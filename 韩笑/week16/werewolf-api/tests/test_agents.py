"""角色 Agent 单元测试 — 创建 / 接口一致性 / 规则兜底"""

from __future__ import annotations

import pytest

from app.agents.werewolf_agent import WerewolfAgent
from app.agents.seer_agent import SeerAgent, VerifiedList
from app.agents.witch_agent import WitchAgent
from app.agents.villager_agent import VillagerAgent
from app.schemas.actions import AgentDecision, SuspectLevel
from app.schemas.messages import Phase, Role


# ============================================================
# Agent 工厂
# ============================================================

AGENT_CLASSES = {
    Role.WEREWOLF: WerewolfAgent,
    Role.SEER: SeerAgent,
    Role.WITCH: WitchAgent,
    Role.VILLAGER: VillagerAgent,
}


def create_agent(role: Role, player_id: int, name: str = "", style: str = "balanced", llm_adapter=None):
    """Agent 工厂函数"""
    cls = AGENT_CLASSES.get(role)
    if cls is None:
        raise ValueError(f"未知角色: {role}")
    return cls(player_id=player_id, name=name or role.value, style=style, llm_adapter=llm_adapter)


# ============================================================
# 工厂测试
# ============================================================

class TestAgentFactory:
    def test_create_all_roles(self):
        for role in (Role.WEREWOLF, Role.SEER, Role.WITCH, Role.VILLAGER):
            agent = create_agent(role, player_id=0)
            assert agent.ROLE == role

    def test_create_invalid_role_raises(self):
        with pytest.raises(ValueError, match="未知角色"):
            create_agent("invalid", player_id=0)  # type: ignore

    def test_agent_has_player_id(self):
        agent = create_agent(Role.WEREWOLF, player_id=3, name="狼A")
        assert agent.player_id == 3
        assert agent.name == "狼A"


# ============================================================
# WerewolfAgent
# ============================================================

class TestWerewolfAgent:
    @pytest.fixture
    def agent(self) -> WerewolfAgent:
        return WerewolfAgent(player_id=0, name="狼A")

    def test_role(self, agent):
        assert agent.ROLE == Role.WEREWOLF

    def test_initial_alive(self, agent):
        assert agent.is_alive is True

    def test_private_info_teammates(self, agent):
        from app.schemas.messages import PrivateInfoMessage
        msg = PrivateInfoMessage(info_type="werewolf_info", payload={"teammates": [1, 3]})
        import asyncio
        asyncio.run(agent._handle_private_info(msg))
        assert agent._teammates == [1, 3]

    def test_rule_based_night_kill(self, agent):
        context = {
            "my_id": 0, "my_name": "狼A", "my_style": "balanced",
            "teammates": [2], "phase": "night_wolf", "round": 1,
            "valid_actions": ["kill_1", "kill_3", "skip"],
            "alive_players": [0, 1, 2, 3],
        }
        decision = agent._rule_based_decide(context, Phase.NIGHT_WOLF, context["valid_actions"])
        assert decision.action == "kill"
        assert decision.target in [1, 3]  # 非队友

    def test_rule_based_night_skip_when_no_target(self, agent):
        agent._teammates = [2]  # 先设置队友
        context = {
            "my_id": 0, "my_name": "狼A", "my_style": "balanced",
            "teammates": [2], "phase": "night_wolf", "round": 1,
            "valid_actions": ["skip"],
            "alive_players": [0, 2],  # 只有自己和队友
        }
        decision = agent._rule_based_decide(context, Phase.NIGHT_WOLF, context["valid_actions"])
        assert decision.action == "skip"

    def test_rule_based_speech(self, agent):
        context = {
            "my_id": 0, "my_name": "狼A", "my_style": "balanced",
            "teammates": [2], "phase": "speech", "round": 1,
            "valid_actions": ["speak"],
            "alive_players": [0, 1, 2, 3],
        }
        decision = agent._rule_based_decide(context, Phase.SPEECH, context["valid_actions"])
        assert decision.action == "speak"
        assert "平民" in decision.content or "玩家" in decision.content

    def test_rule_based_vote_non_teammate(self, agent):
        context = {
            "my_id": 0, "my_name": "狼A", "my_style": "balanced",
            "teammates": [2], "phase": "vote", "round": 1,
            "valid_actions": ["vote_1", "vote_2", "vote_3", "abstain"],
            "alive_players": [0, 1, 2, 3],
        }
        decision = agent._rule_based_decide(context, Phase.VOTE, context["valid_actions"])
        assert decision.action == "vote"
        assert decision.target != 2  # 不投队友
        assert decision.target != 0  # 不投自己


# ============================================================
# SeerAgent
# ============================================================

class TestSeerAgent:
    @pytest.fixture
    def agent(self) -> SeerAgent:
        return SeerAgent(player_id=1, name="预言家")

    def test_role(self, agent):
        assert agent.ROLE == Role.SEER

    def test_verified_list_initial_empty(self, agent):
        assert agent._verified.checked_count == 0

    def test_verified_list_record(self):
        vl = VerifiedList()
        vl.record(0, True)
        vl.record(2, False)
        assert vl.is_wolf(0) is True
        assert vl.is_wolf(2) is False
        assert vl.gold_water == [2]
        assert vl.wolf_check == [0]
        assert vl.checked_count == 2

    def test_verified_list_to_context(self):
        vl = VerifiedList()
        vl.record(3, False)
        ctx = vl.to_context()
        assert 3 in ctx["verified"]
        assert ctx["gold_water"] == [3]
        assert ctx["wolf_check"] == []

    def test_private_info_verify_result(self, agent):
        from app.schemas.messages import PrivateInfoMessage
        # GM 传递的格式: {"verified": {pid_str: is_wolf}}
        msg = PrivateInfoMessage(
            info_type="seer_info",
            payload={"verified": {"3": True}},
        )
        import asyncio
        asyncio.run(agent._handle_private_info(msg))
        assert agent._verified.is_wolf(3) is True

    def test_rule_based_night_verify_unchecked(self, agent):
        agent._verified.record(0, False)
        context = {
            "my_id": 1, "my_name": "预言家", "my_style": "balanced",
            "phase": "night_seer", "round": 2,
            "valid_actions": ["verify_0", "verify_2", "verify_3"],
            "alive_players": [0, 1, 2, 3],
            "unchecked": [2, 3],
            "gold_water": [0], "wolf_check": [],
            "verified": {0: False}, "checked_count": 1,
            "revealed_info": [],
        }
        decision = agent._rule_based_decide(context, Phase.NIGHT_SEER, context["valid_actions"])
        assert decision.action == "verify"
        assert decision.target in [2, 3]  # 只查验未查过的

    def test_rule_based_speech_reports_results(self, agent):
        agent._verified.record(3, True)
        context = {
            "my_id": 1, "my_name": "预言家", "my_style": "balanced",
            "phase": "speech", "round": 2,
            "valid_actions": ["speak"],
            "alive_players": [0, 1, 2, 3, 4, 5],
            "gold_water": [], "wolf_check": [3],
            "verified": {3: True}, "checked_count": 1,
            "revealed_info": [],
            "unchecked": [0, 2, 4, 5],
        }
        decision = agent._rule_based_decide(context, Phase.SPEECH, context["valid_actions"])
        assert decision.action == "speak"
        assert "预言家" in decision.content
        assert "查杀" in decision.content


# ============================================================
# WitchAgent
# ============================================================

class TestWitchAgent:
    @pytest.fixture
    def agent(self) -> WitchAgent:
        return WitchAgent(player_id=2, name="女巫")

    def test_role(self, agent):
        assert agent.ROLE == Role.WITCH

    def test_initial_antidote_available(self, agent):
        assert agent._has_antidote is True

    def test_initial_poison_unavailable(self, agent):
        assert agent._has_poison is False  # 6人板默认无毒药

    def test_first_night_self_save(self, agent):
        context = {
            "my_id": 2, "my_name": "女巫", "my_style": "balanced",
            "phase": "night_witch", "round": 1,
            "valid_actions": ["save", "nosave"],
            "alive_players": [0, 1, 2, 3, 4, 5],
            "has_antidote": True, "has_poison": False,
            "night_kill_target": 2,  # 自己被刀
            "is_first_night": True,
        }
        decision = agent._rule_based_decide(context, Phase.NIGHT_WITCH, context["valid_actions"])
        assert decision.action == "save"
        assert decision.target == 2

    def test_save_other_first_night(self, agent):
        context = {
            "my_id": 2, "my_name": "女巫", "my_style": "balanced",
            "phase": "night_witch", "round": 1,
            "valid_actions": ["save", "nosave"],
            "alive_players": [0, 1, 2, 3, 4, 5],
            "has_antidote": True, "has_poison": False,
            "night_kill_target": 3,
            "is_first_night": True,
        }
        decision = agent._rule_based_decide(context, Phase.NIGHT_WITCH, context["valid_actions"])
        assert decision.action == "save"
        assert decision.target == 3

    def test_no_save_if_antidote_used(self, agent):
        agent._has_antidote = False
        context = {
            "my_id": 2, "my_name": "女巫", "my_style": "balanced",
            "phase": "night_witch", "round": 2,
            "valid_actions": ["nosave"],
            "alive_players": [0, 1, 2, 3],
            "has_antidote": False, "has_poison": False,
            "night_kill_target": 1,
            "is_first_night": False,
        }
        decision = agent._rule_based_decide(context, Phase.NIGHT_WITCH, context["valid_actions"])
        assert decision.action == "nosave"

    def test_speech_hides_identity(self, agent):
        context = {
            "my_id": 2, "my_name": "女巫", "my_style": "balanced",
            "phase": "speech", "round": 1,
            "valid_actions": ["speak"],
            "alive_players": [0, 1, 2, 3, 4, 5],
        }
        decision = agent._rule_based_decide(context, Phase.SPEECH, context["valid_actions"])
        assert decision.action == "speak"
        # 不应暴露女巫身份
        assert "女巫" not in decision.content


# ============================================================
# VillagerAgent
# ============================================================

class TestVillagerAgent:
    @pytest.fixture
    def agent(self) -> VillagerAgent:
        return VillagerAgent(player_id=4, name="平民A")

    def test_role(self, agent):
        assert agent.ROLE == Role.VILLAGER

    def test_private_info_noop(self, agent):
        from app.schemas.messages import PrivateInfoMessage
        msg = PrivateInfoMessage(info_type="test", payload={"data": "secret"})
        import asyncio
        asyncio.run(agent._handle_private_info(msg))
        # 平民不处理私有信息，无副作用

    def test_set_suspicion(self, agent):
        agent.set_suspicion(0, SuspectLevel.LIKELY_WOLF)
        assert agent._suspicions[0] == SuspectLevel.LIKELY_WOLF

    def test_get_highest_suspect(self, agent):
        agent.set_suspicion(1, SuspectLevel.LIKELY_WOLF)
        agent.set_suspicion(2, SuspectLevel.CONFIRMED_WOLF)
        result = agent._get_highest_suspect([0, 1, 2, 3])
        assert result == 2  # CONFIRMED_WOLF > LIKELY_WOLF

    def test_get_highest_suspect_only_dead(self, agent):
        agent.set_suspicion(0, SuspectLevel.CONFIRMED_WOLF)
        result = agent._get_highest_suspect([1, 2, 3])  # 0不在存活列表
        assert result is None

    def test_rule_based_speech_analysis(self, agent):
        context = {
            "my_id": 4, "my_name": "平民A", "my_style": "balanced",
            "phase": "speech", "round": 1,
            "valid_actions": ["speak"],
            "alive_players": [0, 1, 2, 3, 4, 5],
            "suspicions": {},
            "observed_claims": {1: "预言家"},
        }
        decision = agent._rule_based_decide(context, Phase.SPEECH, context["valid_actions"])
        assert decision.action == "speak"
        assert "平民" in decision.content

    def test_rule_based_vote_abstain_no_suspect(self, agent):
        context = {
            "my_id": 4, "my_name": "平民A", "my_style": "balanced",
            "phase": "vote", "round": 1,
            "valid_actions": ["vote_0", "vote_1", "abstain"],
            "alive_players": [0, 1, 2, 3, 4, 5],
        }
        decision = agent._rule_based_decide(context, Phase.VOTE, context["valid_actions"])
        assert decision.action == "abstain"

    def test_rule_based_vote_highest_suspect(self, agent):
        agent.set_suspicion(0, SuspectLevel.LIKELY_WOLF)
        context = {
            "my_id": 4, "my_name": "平民A", "my_style": "balanced",
            "phase": "vote", "round": 2,
            "valid_actions": ["vote_0", "vote_1", "abstain"],
            "alive_players": [0, 1, 2, 3, 4, 5],
        }
        decision = agent._rule_based_decide(context, Phase.VOTE, context["valid_actions"])
        assert decision.action == "vote"
        assert decision.target == 0
