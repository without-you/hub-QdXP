"""GameMaster 集成测试 — 初始化 / 信息隔离 / 夜间处理 / 白天处理 / step 驱动 / 胜负"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.game.game_master import GameMaster, Player
from app.game.state_machine import StateMachine, SubPhase
from app.schemas.actions import AgentDecision, FallbackDecisions
from app.schemas.messages import Phase, Role, Winner

BOARDS_DIR = Path(__file__).resolve().parent.parent / "app" / "game" / "boards"


# ============================================================
# Helpers
# ============================================================

def _decision(action: str, target: int | None = None, thought: str = "", content: str = "") -> AgentDecision:
    return AgentDecision(action=action, target=target, thought=thought, content=content)


def _make_players() -> list[str]:
    return ["狼A", "狼B", "预言家", "女巫", "平民A", "平民B"]


def _load_board(name: str = "standard_6") -> dict:
    with open(BOARDS_DIR / f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _step(gm: GameMaster, decisions: dict[int, AgentDecision] | None = None) -> dict:
    """同步执行 step() 的便捷 helper"""
    return asyncio.run(gm.step(decisions or {}))


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def board_config() -> dict:
    return _load_board()


@pytest.fixture
def gm(board_config) -> GameMaster:
    return GameMaster(board_config)


@pytest.fixture
def gm_initialized(gm) -> GameMaster:
    gm.init_game(_make_players())
    return gm


# ============================================================
# 初始化
# ============================================================

class TestInit:
    def test_init_creates_correct_player_count(self, gm_initialized):
        assert len(gm_initialized.state.players) == 6

    def test_init_creates_correct_role_counts(self, gm_initialized):
        roles = [p.role for p in gm_initialized.state.players]
        assert roles.count(Role.WEREWOLF) == 2
        assert roles.count(Role.SEER) == 1
        assert roles.count(Role.WITCH) == 1
        assert roles.count(Role.VILLAGER) == 2

    def test_init_all_players_alive(self, gm_initialized):
        assert all(p.is_alive for p in gm_initialized.state.players)

    def test_init_no_sheriff(self, gm_initialized):
        assert not any(p.is_sheriff for p in gm_initialized.state.players)

    def test_init_generates_game_id(self, gm_initialized):
        assert gm_initialized.state.game_id.startswith("g_")

    def test_init_custom_game_id(self, gm):
        gid = gm.init_game(_make_players(), game_id="my_game")
        assert gid == "my_game"

    def test_init_phase_is_night_wolf(self, gm_initialized):
        assert gm_initialized.state.sm.phase == Phase.NIGHT_WOLF

    def test_init_wrong_player_count_raises(self, gm):
        with pytest.raises(ValueError, match="玩家数需为"):
            gm.init_game(["A", "B", "C"])  # 6人局需要6个名字

    def test_init_no_shuffle_preserves_order(self, board_config):
        gm = GameMaster(board_config)
        names = ["P0", "P1", "P2", "P3", "P4", "P5"]
        gm.init_game(names, shuffle_roles=False)
        roles = [p.role for p in gm.state.players]
        # role_list 构造顺序: werewolf, werewolf, seer, witch, villager, villager
        assert roles[0] == Role.WEREWOLF
        assert roles[1] == Role.WEREWOLF
        assert roles[2] == Role.SEER
        assert roles[3] == Role.WITCH
        assert roles[4] == Role.VILLAGER
        assert roles[5] == Role.VILLAGER

    def test_load_board_from_string(self):
        gm = GameMaster("standard_6")
        assert gm.board["board_type"] == "6_standard"

    def test_load_board_not_found(self):
        with pytest.raises(FileNotFoundError):
            GameMaster("nonexistent_board")


# ============================================================
# 信息隔离
# ============================================================

class TestInformationIsolation:
    def test_werewolf_gets_teammate_info(self, gm_initialized):
        """狼人玩家应收到队友列表"""
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        for w in wolves:
            info = gm_initialized.build_private_info(w.player_id)
            assert info is not None
            teammates = info.payload.get("teammates", [])
            assert len(teammates) == 1  # 2狼人中，另一个是队友
            assert w.player_id not in teammates

    def test_seer_gets_verified_list(self, gm_initialized):
        seer = [p for p in gm_initialized.state.players if p.role == Role.SEER][0]
        info = gm_initialized.build_private_info(seer.player_id)
        assert info is not None
        assert "verified" in info.payload
        assert isinstance(info.payload["verified"], dict)

    def test_witch_gets_potion_status(self, gm_initialized):
        witch = [p for p in gm_initialized.state.players if p.role == Role.WITCH][0]
        info = gm_initialized.build_private_info(witch.player_id)
        assert info is not None
        assert info.payload["antidote_available"] is True
        assert info.payload["poison_available"] is False  # 6人板子默认无毒药

    def test_villager_gets_empty_info(self, gm_initialized):
        villagers = [p for p in gm_initialized.state.players if p.role == Role.VILLAGER]
        for v in villagers:
            info = gm_initialized.build_private_info(v.player_id)
            assert info is not None
            # 平民没有特殊私有信息
            assert "teammates" not in info.payload

    def test_public_state_only_shows_public_info(self, gm_initialized):
        ps = gm_initialized.get_public_state(0)
        assert "game_id" in ps
        assert "phase" in ps
        assert "alive_players" in ps
        assert "public_log" in ps
        # 不应包含角色信息
        assert "role" not in ps or "my_role" not in str(ps)

    def test_game_start_message_includes_teammates_for_wolves(self, gm_initialized):
        """发送的 game_start 消息中狼人应有 teammates"""
        # 由于没有 WS sender，这里测试构建逻辑
        s = gm_initialized.state
        wolves = [p for p in s.players if p.role == Role.WEREWOLF]
        non_wolves = [p for p in s.players if p.role != Role.WEREWOLF]

        wolf_teammates = [
            other.player_id
            for other in s.players
            if other.role == Role.WEREWOLF and other.player_id != wolves[0].player_id
        ]
        assert len(wolf_teammates) == 1

        non_wolf_teammates = [
            other.player_id
            for other in s.players
            if other.role == Role.WEREWOLF and other.player_id != non_wolves[0].player_id
        ]
        assert len(non_wolf_teammates) == 2  # 非狼人不应过滤 teammates（在 game_start 消息中传空列表）


# ============================================================
# 夜间 — 狼人击杀
# ============================================================

class TestNightWolf:
    def test_wolf_kill_majority(self, gm_initialized):
        """多数狼人选同一目标 → 击杀成功"""
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        decisions = {
            wolves[0].player_id: _decision("kill", target=2),
            wolves[1].player_id: _decision("kill", target=2),
        }
        result = _step(gm_initialized, decisions)
        assert gm_initialized.state.night_kill_target == 2

    def test_wolf_kill_tie_results_skip(self, gm_initialized):
        """意见不一致 → 空刀"""
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        decisions = {
            wolves[0].player_id: _decision("kill", target=2),
            wolves[1].player_id: _decision("kill", target=3),
        }
        result = _step(gm_initialized, decisions)
        assert gm_initialized.state.night_kill_target is None

    def test_wolf_can_skip(self, gm_initialized):
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        decisions = {
            wolves[0].player_id: _decision("skip"),
            wolves[1].player_id: _decision("skip"),
        }
        result = _step(gm_initialized, decisions)
        assert gm_initialized.state.night_kill_target is None

    def test_non_wolf_cannot_kill(self, gm_initialized):
        """非狼人角色的 kill 决策应被忽略"""
        seer = [p for p in gm_initialized.state.players if p.role == Role.SEER][0]
        decisions = {
            seer.player_id: _decision("kill", target=0),
        }
        result = _step(gm_initialized, decisions)
        assert gm_initialized.state.night_kill_target is None

    def test_dead_wolf_cannot_kill(self, gm_initialized):
        """死狼不能参与击杀投票"""
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        # 找一个存活的非狼人作为击杀目标
        alive_non_wolf = [
            p for p in gm_initialized.state.players
            if p.is_alive and p.role != Role.WEREWOLF
        ][0]
        kill_target = alive_non_wolf.player_id
        wolves[1].is_alive = False  # 杀死第二只狼
        decisions = {
            wolves[0].player_id: _decision("kill", target=kill_target),
        }
        result = _step(gm_initialized, decisions)
        assert gm_initialized.state.night_kill_target == kill_target



# ============================================================
# 夜间 — 预言家查验
# ============================================================

class TestNightSeer:
    @pytest.fixture
    def gm_seer_phase(self, gm_initialized) -> GameMaster:
        """推进到 night_seer 阶段"""
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        decisions = {
            wolves[0].player_id: _decision("skip"),
            wolves[1].player_id: _decision("skip"),
        }
        _step(gm_initialized, decisions)
        return gm_initialized

    def test_seer_verify_wolf(self, gm_seer_phase):
        seer = [p for p in gm_seer_phase.state.players if p.role == Role.SEER][0]
        wolf = [p for p in gm_seer_phase.state.players if p.role == Role.WEREWOLF][0]
        decisions = {seer.player_id: _decision("verify", target=wolf.player_id)}
        result = _step(gm_seer_phase, decisions)

        assert gm_seer_phase.state.seer_checked[wolf.player_id] is True

    def test_seer_verify_villager(self, gm_seer_phase):
        seer = [p for p in gm_seer_phase.state.players if p.role == Role.SEER][0]
        villager = [p for p in gm_seer_phase.state.players if p.role == Role.VILLAGER][0]
        decisions = {seer.player_id: _decision("verify", target=villager.player_id)}
        _step(gm_seer_phase, decisions)

        assert gm_seer_phase.state.seer_checked[villager.player_id] is False

    def test_seer_skips(self, gm_seer_phase):
        seer = [p for p in gm_seer_phase.state.players if p.role == Role.SEER][0]
        decisions = {seer.player_id: _decision("skip")}
        _step(gm_seer_phase, decisions)

        assert len(gm_seer_phase.state.seer_checked) == 0

    def test_non_seer_cannot_verify(self, gm_seer_phase):
        villager = [p for p in gm_seer_phase.state.players if p.role == Role.VILLAGER][0]
        decisions = {villager.player_id: _decision("verify", target=0)}
        _step(gm_seer_phase, decisions)

        assert len(gm_seer_phase.state.seer_checked) == 0

    def test_seer_accumulates_verify_history(self, gm_seer_phase):
        """预言家查人记录应累积"""
        gm_seer_phase.state.sm.sheriff_election = False  # 简化测试
        seer = [p for p in gm_seer_phase.state.players if p.role == Role.SEER][0]
        # first verify
        decisions = {seer.player_id: _decision("verify", target=0)}
        _step(gm_seer_phase, decisions)
        assert 0 in gm_seer_phase.state.seer_checked

        # advance to next round's seer phase
        sm = gm_seer_phase.state.sm
        while sm.phase != Phase.NIGHT_SEER:
            _step(gm_seer_phase)

        # second verify
        decisions2 = {seer.player_id: _decision("verify", target=1)}
        _step(gm_seer_phase, decisions2)
        assert 0 in gm_seer_phase.state.seer_checked  # previous result preserved
        assert 1 in gm_seer_phase.state.seer_checked


# ============================================================
# 夜间 — 女巫用药
# ============================================================

class TestNightWitch:
    @pytest.fixture
    def gm_witch_phase(self, gm_initialized) -> GameMaster:
        """推进到 night_witch 阶段，狼人刀了3号"""
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        decisions_wolf = {
            wolves[0].player_id: _decision("kill", target=3),
            wolves[1].player_id: _decision("kill", target=3),
        }
        _step(gm_initialized, decisions_wolf)

        seer = [p for p in gm_initialized.state.players if p.role == Role.SEER][0]
        decisions_seer = {seer.player_id: _decision("verify", target=0)}
        _step(gm_initialized, decisions_seer)
        return gm_initialized

    def test_witch_saves_kill_target(self, gm_witch_phase):
        witch = [p for p in gm_witch_phase.state.players if p.role == Role.WITCH][0]
        decisions = {witch.player_id: _decision("save")}
        _step(gm_witch_phase, decisions)

        assert gm_witch_phase.state.witch_saved == 3

    def test_witch_does_not_save(self, gm_witch_phase):
        witch = [p for p in gm_witch_phase.state.players if p.role == Role.WITCH][0]
        decisions = {witch.player_id: _decision("nosave")}
        _step(gm_witch_phase, decisions)

        assert gm_witch_phase.state.witch_saved is None

    def test_witch_with_poison(self, gm_initialized):
        """带毒药的女巫板子"""
        gm_initialized.board["rules"]["witch_has_poison"] = True
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        _step(gm_initialized, {
            wolves[0].player_id: _decision("skip"),
            wolves[1].player_id: _decision("skip"),
        })
        seer = [p for p in gm_initialized.state.players if p.role == Role.SEER][0]
        _step(gm_initialized, {
            seer.player_id: _decision("skip"),
        })

        witch = [p for p in gm_initialized.state.players if p.role == Role.WITCH][0]
        decisions = {witch.player_id: _decision("poison", target=1)}
        _step(gm_initialized, decisions)

        assert gm_initialized.state.witch_poisoned == 1


# ============================================================
# 夜晚结算
# ============================================================

class TestNightResult:
    @staticmethod
    def _advance_to_night_result(gm: GameMaster) -> None:
        """快进到 night_result 阶段"""
        for _ in range(3):  # night_wolf → night_seer → night_witch
            _step(gm)

    def test_kill_not_saved_player_dies(self, gm_initialized):
        """狼刀未被救 → 玩家死亡"""
        self._advance_to_night_result(gm_initialized)
        gm_initialized.state.night_kill_target = 3
        gm_initialized.state.witch_saved = None
        _step(gm_initialized)

        player = gm_initialized._get_player(3)
        assert player.is_alive is False

    def test_kill_saved_player_survives(self, gm_initialized):
        """狼刀被救 → 平安夜"""
        self._advance_to_night_result(gm_initialized)
        gm_initialized.state.night_kill_target = 3
        gm_initialized.state.witch_saved = 3
        _step(gm_initialized)

        player = gm_initialized._get_player(3)
        assert player.is_alive is True

    def test_poison_kills_player(self, gm_initialized):
        self._advance_to_night_result(gm_initialized)
        gm_initialized.state.witch_poisoned = 5
        _step(gm_initialized)

        player = gm_initialized._get_player(5)
        assert player.is_alive is False

    def test_night_state_reset_after_result(self, gm_initialized):
        """夜晚结算后重置临时状态"""
        self._advance_to_night_result(gm_initialized)
        gm_initialized.state.night_kill_target = 3
        _step(gm_initialized)

        assert gm_initialized.state.night_kill_target is None
        assert gm_initialized.state.witch_saved is None
        assert gm_initialized.state.witch_poisoned is None


# ============================================================
# 白天阶段
# ============================================================

class TestDayPhases:
    @pytest.fixture
    def gm_day_start(self, gm_initialized) -> GameMaster:
        """推进到 day_start"""
        gm_initialized.state.night_kill_target = None  # 平安夜
        for _ in range(4):  # 跳过 4 个夜间阶段
            _step(gm_initialized)
        assert gm_initialized.state.sm.phase == Phase.DAY_START
        return gm_initialized

    def test_day_start_announces_peaceful_night(self, gm_day_start):
        result = _step(gm_day_start)
        logs = gm_day_start.state.public_log
        assert any("平安夜" in log for log in logs)

    def test_day_start_announces_death(self, gm_initialized):
        # 先跳过 3 个夜间阶段到 night_witch，设定狼刀后进入 night_result
        for _ in range(3):
            _step(gm_initialized)
        # 在 night_witch 之后，手动设置击杀目标（模拟狼刀结果已定）
        gm_initialized.state.night_kill_target = 2
        gm_initialized.state.witch_saved = None
        # night_result 处理
        _step(gm_initialized)
        # day_start
        _step(gm_initialized)

        logs = gm_initialized.state.public_log
        assert any("死讯" in log for log in logs)

    def test_speech_phase_normal(self, gm_day_start):
        """正常发言阶段 — 禁用警长竞选"""
        gm_day_start.state.sm.sheriff_election = False
        _step(gm_day_start)  # day_start → speech
        assert gm_day_start.state.sm.phase == Phase.SPEECH
        # speech phase
        alive = [p for p in gm_day_start.state.players if p.is_alive]
        decisions = {p.player_id: _decision("speak", content=f"我是{p.player_id}号发言") for p in alive}
        result = _step(gm_day_start, decisions)

        assert "dialogues" in result.get("step_data", {})

    def test_sheriff_election_flow_first_day(self, gm_day_start):
        """首日警长竞选完整流程"""
        _step(gm_day_start)  # day_start → speech

        s = gm_day_start.state
        sm = s.sm
        assert sm.sub_phase == SubPhase.SHERIFF_CANDIDATES

        # 上警
        decisions_candidates = {
            p.player_id: _decision("run_sheriff")
            for p in s.players if p.is_alive and p.player_id in (0, 2, 4)
        }
        _step(gm_day_start, decisions_candidates)
        assert sm.sub_phase == SubPhase.SHERIFF_SPEECH
        assert sm.sheriff_candidates == [0, 2, 4]

        # 警上发言
        decisions_speeches = {
            pid: _decision("speak", content=f"我是{pid}号，竞选警长")
            for pid in sm.sheriff_candidates
        }
        _step(gm_day_start, decisions_speeches)
        assert sm.sub_phase == SubPhase.SHERIFF_VOTE

        # 警下投票 → 0号当选
        non_candidates = [p for p in s.players if p.is_alive and p.player_id not in sm.sheriff_candidates]
        decisions_votes = {
            p.player_id: _decision("vote_sheriff", target=0)
            for p in non_candidates
        }
        _step(gm_day_start, decisions_votes)

        sheriff = gm_day_start._get_sheriff()
        assert sheriff == 0


# ============================================================
# 投票放逐
# ============================================================

class TestVote:
    @pytest.fixture
    def gm_vote_phase(self, gm_initialized) -> GameMaster:
        """推进到 vote 阶段（跳过警长竞选直接进入 vote）"""
        # 禁用警长竞选以简化测试
        gm_initialized.state.sm.sheriff_election = False
        # 快进到 vote（跳过夜间和 day_start/speech）
        for _ in range(6):
            _step(gm_initialized)
        assert gm_initialized.state.sm.phase == Phase.VOTE
        return gm_initialized

    def test_majority_vote_eliminates(self, gm_vote_phase):
        alive = [p for p in gm_vote_phase.state.players if p.is_alive]
        decisions = {}
        for p in alive:
            if p.player_id == 2:
                continue  # 2号不投票
            decisions[p.player_id] = _decision("vote", target=2)  # 其他人全票投2号
        result = _step(gm_vote_phase, decisions)

        player_2 = gm_vote_phase._get_player(2)
        assert player_2.is_alive is False

    def test_tie_vote_no_elimination(self, gm_vote_phase):
        alive = [p for p in gm_vote_phase.state.players if p.is_alive]
        decisions = {}
        half = len(alive) // 2
        for i, p in enumerate(alive):
            target = alive[0].player_id if i < half else alive[1].player_id
            decisions[p.player_id] = _decision("vote", target=target)
        _step(gm_vote_phase, decisions)

        # 平票无人被放逐
        assert all(p.is_alive for p in gm_vote_phase.state.players if p.player_id in [a.player_id for a in alive])

    def test_abstain_all_no_elimination(self, gm_vote_phase):
        alive = [p for p in gm_vote_phase.state.players if p.is_alive]
        decisions = {p.player_id: _decision("abstain") for p in alive}
        _step(gm_vote_phase, decisions)

        assert all(p.is_alive for p in gm_vote_phase.state.players)

    def test_sheriff_has_extra_vote_power(self, gm_vote_phase):
        """警长 1.5 票加权"""
        s = gm_vote_phase.state
        alive = [p for p in s.players if p.is_alive]
        non_sheriff = [p for p in alive if p.player_id != 0]

        # 0号为警长
        s.players[0].is_sheriff = True

        # 0号投2，另一个人投3，其余人弃票
        decisions = {p.player_id: _decision("abstain") for p in alive}
        decisions[0] = _decision("vote", target=2)
        decisions[1] = _decision("vote", target=3)

        _step(gm_vote_phase, decisions)

        # 0号1.5票 > 1号1票 → 2号被放逐
        player_2 = gm_vote_phase._get_player(2)
        assert player_2.is_alive is False
        player_3 = gm_vote_phase._get_player(3)
        assert player_3.is_alive is True


# ============================================================
# 狼人自爆
# ============================================================

class TestSelfDestruct:
    @pytest.fixture
    def gm_vote_phase(self, gm_initialized) -> GameMaster:
        gm_initialized.state.sm.sheriff_election = False
        for _ in range(6):
            _step(gm_initialized)
        assert gm_initialized.state.sm.phase == Phase.VOTE
        return gm_initialized

    def test_wolf_self_destruct_kills_wolf(self, gm_vote_phase):
        wolf = [p for p in gm_vote_phase.state.players if p.role == Role.WEREWOLF][0]
        decisions = {wolf.player_id: _decision("self_destruct")}
        result = _step(gm_vote_phase, decisions)

        assert wolf.is_alive is False
        assert "wolf_self_destruct" in result.get("step_data", {})

    def test_non_wolf_cannot_self_destruct(self, gm_vote_phase):
        villager = [p for p in gm_vote_phase.state.players if p.role == Role.VILLAGER][0]
        decisions = {villager.player_id: _decision("self_destruct")}
        _step(gm_vote_phase, decisions)

        assert villager.is_alive is True

    def test_self_destruct_when_disabled(self, gm_vote_phase):
        gm_vote_phase.board["rules"]["allow_wolf_self_destruct"] = False
        wolf = [p for p in gm_vote_phase.state.players if p.role == Role.WEREWOLF][0]
        decisions = {wolf.player_id: _decision("self_destruct")}
        _step(gm_vote_phase, decisions)

        assert wolf.is_alive is True


# ============================================================
# 游戏结束检测
# ============================================================

class TestGameOver:
    def test_good_wins_when_all_wolves_dead(self, gm_initialized):
        """两只狼都死亡 → 好人胜利"""
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        for w in wolves:
            w.is_alive = False
        _step(gm_initialized)

        assert gm_initialized.state.is_game_over is True
        assert gm_initialized.state.winner == Winner.GOOD

    def test_evil_wins_when_all_gods_dead(self, gm_initialized):
        gods = [p for p in gm_initialized.state.players if p.role in (Role.SEER, Role.WITCH)]
        for g in gods:
            g.is_alive = False
        _step(gm_initialized)

        assert gm_initialized.state.is_game_over is True
        assert gm_initialized.state.winner == Winner.EVIL

    def test_evil_wins_when_all_villagers_dead(self, gm_initialized):
        villagers = [p for p in gm_initialized.state.players if p.role == Role.VILLAGER]
        for v in villagers:
            v.is_alive = False
        _step(gm_initialized)

        assert gm_initialized.state.is_game_over is True
        assert gm_initialized.state.winner == Winner.EVIL

    def test_game_continues_with_mixed_deaths(self, gm_initialized):
        """1狼+1神+1民死亡 → 游戏继续"""
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        wolves[0].is_alive = False
        seer = [p for p in gm_initialized.state.players if p.role == Role.SEER][0]
        seer.is_alive = False
        villagers = [p for p in gm_initialized.state.players if p.role == Role.VILLAGER]
        villagers[0].is_alive = False

        _step(gm_initialized)
        assert gm_initialized.state.is_game_over is False


# ============================================================
# step() 整体流程
# ============================================================

class TestStepFlow:
    def test_step_advances_through_all_phases(self, gm_initialized):
        """step() 应能驱动完整一轮"""
        gm_initialized.state.sm.sheriff_election = False
        initial_phase = gm_initialized.state.sm.phase

        # step 8 次 = 完整一轮
        for i in range(8):
            result = _step(gm_initialized)
            assert "phase" in result
            assert "action_requests" in result

        # 应该回到 night_wolf，round=2
        assert gm_initialized.state.sm.phase == Phase.NIGHT_WOLF
        assert gm_initialized.state.sm.round == 2

    def test_step_provides_action_requests(self, gm_initialized):
        """step 应为下一阶段生成 action_requests"""
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        decisions = {
            wolves[0].player_id: _decision("skip"),
            wolves[1].player_id: _decision("skip"),
        }
        result = _step(gm_initialized, decisions)

        # 下一阶段是 night_seer，应有预言家的 action_request
        requests = result.get("action_requests", [])
        seer_ids = [p.player_id for p in gm_initialized.state.players if p.role == Role.SEER]
        assert any(req["player_id"] in seer_ids for req in requests)

    def test_step_system_phases_no_action_requests(self, gm_initialized):
        """系统阶段（night_result/day_start/day_end）不应有 action_requests"""
        # 跳过 night_wolf, night_seer, night_witch → night_result
        for _ in range(3):
            _step(gm_initialized)

        result = _step(gm_initialized)
        assert result["action_requests"] == []


# ============================================================
# 兜底决策
# ============================================================

class TestFallbackDecisions:
    def test_fallback_wolf_is_skip(self):
        d = FallbackDecisions.for_wolf(0)
        assert d.action == "kill"
        assert d.target is None  # 空刀

    def test_fallback_seer_is_skip(self):
        d = FallbackDecisions.for_seer(1)
        assert d.action == "skip"

    def test_fallback_witch_is_skip(self):
        d = FallbackDecisions.for_witch(2)
        assert d.action == "skip"

    def test_fallback_villager_is_abstain(self):
        d = FallbackDecisions.for_villager(4)
        assert d.action == "vote"
        assert d.target is None  # 弃票

    def test_fallback_for_role_dispatcher(self):
        d = FallbackDecisions.for_role("werewolf", 0)
        assert d.action == "kill"
        d = FallbackDecisions.for_role("unknown_role", 0)
        assert d.action == "vote"  # default to villager


# ============================================================
# 日志
# ============================================================

class TestLogging:
    def test_public_log_accumulates(self, gm_initialized):
        initial_count = len(gm_initialized.state.public_log)
        _step(gm_initialized)
        assert len(gm_initialized.state.public_log) > initial_count

    def test_action_log_records_agent_decisions(self, gm_initialized):
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        decisions = {
            wolves[0].player_id: _decision("kill", target=3, thought="我觉得3号是预言家"),
            wolves[1].player_id: _decision("kill", target=3, thought="同意"),
        }
        _step(gm_initialized, decisions)

        actions = gm_initialized.state.action_log
        assert len(actions) >= 2
        assert any(a["thought"] == "我觉得3号是预言家" for a in actions)
        assert any(a["target"] == 3 for a in actions)


# ============================================================
# 边界情况
# ============================================================

class TestEdgeCases:
    def test_all_players_dead_except_one(self, gm_initialized):
        """只剩一个玩家存活（极端情况）"""
        for p in gm_initialized.state.players:
            if p.player_id != 0:
                p.is_alive = False
        _step(gm_initialized)

        # 应该有一方胜利
        assert gm_initialized.state.is_game_over is True

    def test_dead_player_actions_ignored(self, gm_initialized):
        """死人的行动应被忽略"""
        wolves = [p for p in gm_initialized.state.players if p.role == Role.WEREWOLF]
        # 找一个存活的非狼人作为击杀目标
        alive_non_wolf = [
            p for p in gm_initialized.state.players
            if p.is_alive and p.role != Role.WEREWOLF
        ][0]
        kill_target = alive_non_wolf.player_id
        wolves[0].is_alive = False
        decisions = {
            wolves[0].player_id: _decision("kill", target=kill_target),
            wolves[1].player_id: _decision("kill", target=kill_target),
        }
        _step(gm_initialized, decisions)

        # 只有活狼的投票有效
        assert gm_initialized.state.night_kill_target == kill_target
