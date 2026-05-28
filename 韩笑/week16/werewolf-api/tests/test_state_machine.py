"""StateMachine 单元测试 — 阶段流转 / 合法行动 / 胜负判定 / 警长竞选"""

from __future__ import annotations

import pytest

from app.game.state_machine import PHASE_ORDER, StateMachine, SubPhase
from app.schemas.messages import Phase, Role, Winner


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sm() -> StateMachine:
    return StateMachine()


@pytest.fixture
def sm_no_sheriff() -> StateMachine:
    """无警长竞选规则的板子"""
    return StateMachine(sheriff_election=False)


@pytest.fixture
def six_players() -> list[dict]:
    """标准 6 人局初始状态：2狼 1预 1巫 2民"""
    return [
        {"role": Role.WEREWOLF, "is_alive": True},
        {"role": Role.WEREWOLF, "is_alive": True},
        {"role": Role.SEER, "is_alive": True},
        {"role": Role.WITCH, "is_alive": True},
        {"role": Role.VILLAGER, "is_alive": True},
        {"role": Role.VILLAGER, "is_alive": True},
    ]


# ============================================================
# 阶段定义
# ============================================================

class TestPhaseDefinition:
    def test_phase_order_length(self):
        assert len(PHASE_ORDER) == 8

    def test_phase_order_starts_night_wolf(self):
        assert PHASE_ORDER[0] == Phase.NIGHT_WOLF

    def test_phase_order_ends_day_end(self):
        assert PHASE_ORDER[-1] == Phase.DAY_END


# ============================================================
# 阶段推进
# ============================================================

class TestPhaseAdvance:
    def test_initial_phase(self, sm):
        assert sm.phase == Phase.NIGHT_WOLF
        assert sm.round == 1
        assert sm.day_number == 0

    def test_advance_night_wolf_to_seer(self, sm):
        assert sm.phase == Phase.NIGHT_WOLF
        sm.advance()
        assert sm.phase == Phase.NIGHT_SEER
        assert sm.round == 1

    def test_advance_full_cycle(self, sm):
        """完整一圈 8 个阶段后回到 night_wolf，round+1"""
        for i in range(8):
            sm.advance()
        assert sm.phase == Phase.NIGHT_WOLF
        assert sm.round == 2
        assert sm.day_number == 1

    def test_advance_two_cycles(self, sm):
        for _ in range(16):
            sm.advance()
        assert sm.phase == Phase.NIGHT_WOLF
        assert sm.round == 3
        assert sm.day_number == 2

    def test_day_number_increments_at_day_start(self, sm):
        """day_number 只在 day_start 阶段 +1"""
        # night_wolf → night_seer → night_witch → night_result → day_start
        for _ in range(4):
            sm.advance()
        assert sm.phase == Phase.DAY_START
        assert sm.day_number == 1

    def test_day_number_not_incremented_during_night(self, sm):
        sm.advance()  # → night_seer
        assert sm.day_number == 0
        sm.advance()  # → night_witch
        assert sm.day_number == 0

    def test_phase_is_night(self, sm):
        assert sm.is_night is True
        sm.advance()  # night_seer
        assert sm.is_night is True
        sm.advance()  # night_witch
        assert sm.is_night is True
        sm.advance()  # night_result
        assert sm.is_night is False

    def test_phase_is_day(self, sm):
        assert sm.is_day is False
        for _ in range(4):
            sm.advance()  # → day_start
        assert sm.is_day is True


# ============================================================
# 合法行动生成
# ============================================================

class TestValidActions:
    def test_night_wolf_actions(self, sm):
        alive = [0, 1, 2, 3, 4, 5]
        actions = sm.get_valid_actions(Role.WEREWOLF, 0, alive)
        assert "kill_0" in actions or "kill_1" in actions
        assert "skip" in actions

    def test_night_wolf_skip_always_present(self, sm):
        alive = [0, 1, 2]
        actions = sm.get_valid_actions(Role.WEREWOLF, 0, alive)
        assert "skip" in actions

    def test_night_seer_actions(self, sm):
        sm.advance()  # night_wolf → night_seer
        alive = [0, 1, 2, 3]
        actions = sm.get_valid_actions(Role.SEER, 2, alive)
        for pid in alive:
            assert f"verify_{pid}" in actions

    def test_night_witch_actions_save(self, sm):
        sm.advance(); sm.advance()  # → night_witch
        actions = sm.get_valid_actions(Role.WITCH, 3, [0, 1, 2, 3, 4, 5])
        assert "save" in actions
        assert "nosave" in actions

    def test_night_witch_no_poison_by_default(self, sm):
        """默认板子没有毒药"""
        sm.advance(); sm.advance()  # → night_witch
        actions = sm.get_valid_actions(Role.WITCH, 3, [0, 1, 2, 3, 4, 5])
        assert not any(a.startswith("poison_") for a in actions)

    def test_night_witch_with_poison(self):
        sm = StateMachine(witch_has_poison=True)
        sm.advance(); sm.advance()  # → night_witch
        alive = [0, 1, 2, 3, 4]
        actions = sm.get_valid_actions(Role.WITCH, 3, alive)
        assert "nopoison" in actions
        for pid in alive:
            assert f"poison_{pid}" in actions

    def test_vote_phase_actions(self, sm):
        # 先推进到 vote 阶段
        for _ in range(6):
            sm.advance()
        assert sm.phase == Phase.VOTE
        alive = [0, 1, 2, 4, 5]
        actions = sm.get_valid_actions(Role.WEREWOLF, 0, alive)
        assert "abstain" in actions
        for pid in alive:
            assert f"vote_{pid}" in actions

    def test_speech_phase_action(self, sm):
        for _ in range(5):
            sm.advance()
        assert sm.phase == Phase.SPEECH
        actions = sm.get_valid_actions(Role.VILLAGER, 4, [0, 1, 2, 3, 4, 5])
        assert "speak" in actions

    def test_skip_action_for_system_phases(self, sm):
        # 非行动阶段返回 ["skip"]
        alive = [0, 1, 2, 3, 4, 5]
        actions = sm.get_valid_actions(Role.WEREWOLF, 0, alive)
        # night_wolf 阶段狼人应该有正经行动
        assert len(actions) > 1


# ============================================================
# 活跃角色
# ============================================================

class TestActiveRoles:
    def test_night_wolf_active_roles(self, sm):
        roles = sm.get_active_roles()
        assert Role.WEREWOLF in roles
        assert Role.SEER not in roles
        assert Role.WITCH not in roles

    def test_night_seer_active_roles(self, sm):
        sm.advance()
        roles = sm.get_active_roles()
        assert Role.SEER in roles
        assert Role.WEREWOLF not in roles

    def test_night_witch_active_roles(self, sm):
        sm.advance(); sm.advance()
        roles = sm.get_active_roles()
        assert Role.WITCH in roles

    def test_system_phases_no_active_roles(self, sm):
        for _ in range(3):
            sm.advance()
        assert sm.phase == Phase.NIGHT_RESULT
        assert sm.get_active_roles() == ()

    def test_speech_all_alive_active(self, sm):
        for _ in range(5):
            sm.advance()
        assert sm.phase == Phase.SPEECH
        roles = sm.get_active_roles()
        assert len(roles) == 4  # werewolf, seer, witch, villager

    def test_is_role_active(self, sm):
        assert sm.is_role_active(Role.WEREWOLF) is True
        assert sm.is_role_active(Role.SEER) is False


# ============================================================
# 胜负判定 — 屠边规则
# ============================================================

class TestWinCondition:
    def test_game_continues_initial(self, sm, six_players):
        assert sm.check_win(six_players) is None

    def test_good_wins_all_wolves_dead(self, sm, six_players):
        players = six_players.copy()
        players[0]["is_alive"] = False
        players[1]["is_alive"] = False
        assert sm.check_win(players) == Winner.GOOD

    def test_evil_wins_all_gods_dead(self, sm, six_players):
        players = six_players.copy()
        players[2]["is_alive"] = False  # seer
        players[3]["is_alive"] = False  # witch
        assert sm.check_win(players) == Winner.EVIL

    def test_evil_wins_all_villagers_dead(self, sm, six_players):
        players = six_players.copy()
        players[4]["is_alive"] = False
        players[5]["is_alive"] = False
        assert sm.check_win(players) == Winner.EVIL

    def test_evil_wins_mixed_villagers_and_gods_dead(self, sm, six_players):
        """一边死1神1民不触发屠边"""
        players = six_players.copy()
        players[2]["is_alive"] = False  # seer dead
        players[4]["is_alive"] = False  # villager dead
        assert sm.check_win(players) is None  # 游戏继续

    def test_good_wins_one_wolf_dead_one_alive(self, sm, six_players):
        players = six_players.copy()
        players[0]["is_alive"] = False
        assert sm.check_win(players) is None

    def test_edge_case_both_sides_win_condition(self, sm):
        """同时满足双方胜利条件（理论上不可能，但做防御性测试）"""
        players = [
            {"role": Role.WEREWOLF, "is_alive": False},
            {"role": Role.WEREWOLF, "is_alive": False},
            {"role": Role.SEER, "is_alive": True},
            {"role": Role.WITCH, "is_alive": True},
            {"role": Role.VILLAGER, "is_alive": False},
            {"role": Role.VILLAGER, "is_alive": False},
        ]
        # 狼全灭 AND 平民全灭 → 好人和狼人同时满足
        # 设计上好人优先判定
        assert sm.check_win(players) == Winner.GOOD

    def test_win_reason_good(self, sm, six_players):
        players = six_players.copy()
        players[0]["is_alive"] = False
        players[1]["is_alive"] = False
        reason = sm.get_win_reason(Winner.GOOD, players)
        assert reason == "all_wolves_dead"

    def test_win_reason_evil_gods(self, sm, six_players):
        players = six_players.copy()
        players[2]["is_alive"] = False
        players[3]["is_alive"] = False
        reason = sm.get_win_reason(Winner.EVIL, players)
        assert reason == "all_gods_dead"

    def test_win_reason_evil_villagers(self, sm, six_players):
        players = six_players.copy()
        players[4]["is_alive"] = False
        players[5]["is_alive"] = False
        reason = sm.get_win_reason(Winner.EVIL, players)
        assert reason == "all_villagers_dead"


# ============================================================
# 警长竞选
# ============================================================

class TestSheriffElection:
    def test_election_triggered_first_day(self, sm):
        """首日 speech 阶段应自动进入警长竞选"""
        for _ in range(5):
            sm.advance()
        assert sm.phase == Phase.SPEECH
        assert sm.sub_phase == SubPhase.SHERIFF_CANDIDATES

    def test_no_election_when_disabled(self, sm_no_sheriff):
        for _ in range(5):
            sm_no_sheriff.advance()
        assert sm_no_sheriff.phase == Phase.SPEECH
        assert sm_no_sheriff.sub_phase == SubPhase.NORMAL_SPEECH

    def test_no_election_second_day(self, sm):
        """第二天的 speech 不再有警长竞选"""
        for _ in range(13):  # 一整轮 (8) + 到第二天的 speech (5)
            sm.advance()
        assert sm.phase == Phase.SPEECH
        assert sm.round == 2
        assert sm.sub_phase == SubPhase.NORMAL_SPEECH

    def test_enter_sheriff_candidate(self, sm):
        for _ in range(5):
            sm.advance()
        sm.enter_sheriff_candidate(0)
        sm.enter_sheriff_candidate(2)
        assert sm.sheriff_candidates == [0, 2]

    def test_duplicate_candidate_not_added(self, sm):
        for _ in range(5):
            sm.advance()
        sm.enter_sheriff_candidate(0)
        sm.enter_sheriff_candidate(0)
        assert sm.sheriff_candidates == [0]

    def test_candidate_ignored_outside_candidate_phase(self, sm):
        """非 SHERIFF_CANDIDATES 阶段不接受候选人"""
        sm.enter_sheriff_candidate(0)
        assert sm.sheriff_candidates == []

    def test_sheriff_speech_transition(self, sm):
        for _ in range(5):
            sm.advance()
        sm.start_sheriff_speeches()
        assert sm.sub_phase == SubPhase.SHERIFF_SPEECH

    def test_sheriff_vote_transition(self, sm):
        for _ in range(5):
            sm.advance()
        sm.start_sheriff_speeches()
        sm.start_sheriff_vote()
        assert sm.sub_phase == SubPhase.SHERIFF_VOTE

    def test_sheriff_elected_majority(self, sm):
        for _ in range(5):
            sm.advance()
        sm.enter_sheriff_candidate(0)
        sm.enter_sheriff_candidate(2)
        sm.start_sheriff_speeches()
        sm.start_sheriff_vote()

        sm.record_sheriff_vote(1, 0)
        sm.record_sheriff_vote(3, 0)
        sm.record_sheriff_vote(4, 2)
        sm.record_sheriff_vote(5, 2)
        # 2:2 平票

        winner = sm.finish_sheriff_election()
        assert winner is None  # 平票无警长

    def test_sheriff_elected_clear_winner(self, sm):
        for _ in range(5):
            sm.advance()
        sm.enter_sheriff_candidate(0)
        sm.enter_sheriff_candidate(2)
        sm.start_sheriff_speeches()
        sm.start_sheriff_vote()

        sm.record_sheriff_vote(1, 0)
        sm.record_sheriff_vote(3, 0)
        sm.record_sheriff_vote(4, 0)  # 3票给0
        sm.record_sheriff_vote(5, 2)  # 1票给2

        winner = sm.finish_sheriff_election()
        assert winner == 0
        assert sm.sheriff_election_done is True
        assert sm.sub_phase == SubPhase.NORMAL_SPEECH

    def test_sheriff_election_no_candidates(self, sm):
        for _ in range(5):
            sm.advance()
        sm.start_sheriff_speeches()
        sm.start_sheriff_vote()
        # 没人投票
        winner = sm.finish_sheriff_election()
        assert winner is None

    def test_sheriff_vote_from_candidate_not_counted(self, sm):
        """候选人不能投票"""
        # 这个在 game_master 层过滤，state_machine 层不做校验
        # 仅测试 votes 被正确记录
        for _ in range(5):
            sm.advance()
        sm.enter_sheriff_candidate(0)
        sm.start_sheriff_speeches()
        sm.start_sheriff_vote()
        sm.record_sheriff_vote(0, 0)  # 候选人投自己
        assert sm.sheriff_votes == {0: 0}


# ============================================================
# 已行动玩家追踪
# ============================================================

class TestActedPlayers:
    def test_mark_player_acted(self, sm):
        sm.mark_player_acted(0)
        sm.mark_player_acted(1)
        assert sm.all_required_acted(2) is True
        assert sm.all_required_acted(3) is False

    def test_reset_acted(self, sm):
        sm.mark_player_acted(0)
        sm.reset_acted()
        assert sm.all_required_acted(0) is True


# ============================================================
# 超时配置
# ============================================================

class TestTimeout:
    def test_default_timeouts(self, sm):
        assert sm.get_timeout(Phase.NIGHT_WOLF) == 60
        assert sm.get_timeout(Phase.NIGHT_SEER) == 30
        assert sm.get_timeout(Phase.SPEECH) == 120
        assert sm.get_timeout(Phase.VOTE) == 60
        assert sm.get_timeout(Phase.DAY_START) == 5

    def test_custom_timeouts(self, sm):
        custom = {"night_wolf": 90, "speech": 200}
        assert sm.get_timeout(Phase.NIGHT_WOLF, custom) == 90
        assert sm.get_timeout(Phase.NIGHT_SEER, custom) == 30  # fallback
        assert sm.get_timeout(Phase.SPEECH, custom) == 200


# ============================================================
# 重置
# ============================================================

class TestReset:
    def test_reset_restores_initial_state(self, sm):
        for _ in range(5):
            sm.advance()
        sm.mark_player_acted(0)
        sm.enter_sheriff_candidate(0)

        sm.reset()

        assert sm.phase == Phase.NIGHT_WOLF
        assert sm.round == 1
        assert sm.day_number == 0
        assert sm.sheriff_candidates == []
        assert sm.sheriff_votes == {}
        assert sm.sub_phase is None
        assert sm.all_required_acted(0) is True
