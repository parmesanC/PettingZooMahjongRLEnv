import pytest
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.score_calculator import MahjongScoreSettler


class TestCappingLogic:
    """测试封顶和金顶规则"""

    def test_normal_capping_one_over(self):
        """普通封顶：只有一家超过300"""
        settler = MahjongScoreSettler(is_kou_kou_fan=False)
        result = settler._apply_capping([200, 450, 100])
        assert result == [200, 300, 100]

    def test_normal_capping_two_over(self):
        """普通封顶：有两家超过300"""
        settler = MahjongScoreSettler(is_kou_kou_fan=False)
        result = settler._apply_capping([350, 400, 150])
        assert result == [300, 300, 150]

    def test_normal_capping_all_over(self):
        """普通封顶：三家都超过300（触发金顶）"""
        settler = MahjongScoreSettler(is_kou_kou_fan=False)
        result = settler._apply_capping([320, 960, 320])
        assert result == [320, 400, 320]

    def test_golden_cap_normal_mode(self):
        """金顶普通模式：上限400"""
        settler = MahjongScoreSettler(is_kou_kou_fan=False)
        result = settler._apply_capping([500, 800, 600])
        assert result == [400, 400, 400]

    def test_golden_cap_koukou_mode(self):
        """金顶口口翻模式：上限500"""
        settler = MahjongScoreSettler(is_kou_kou_fan=True)
        result = settler._apply_capping([500, 800, 600])
        assert result == [500, 500, 500]

    def test_no_capping_needed(self):
        """无需封顶：所有分数都在300以下"""
        settler = MahjongScoreSettler(is_kou_kou_fan=False)
        result = settler._apply_capping([100, 200, 50])
        assert result == [100, 200, 50]

    def test_boundary_exactly_300(self):
        """边界测试：恰好300"""
        settler = MahjongScoreSettler(is_kou_kou_fan=False)
        result = settler._apply_capping([300, 300, 300])
        assert result == [300, 300, 300]

    def test_golden_cap_mixed_values(self):
        """金顶混合值测试：部分低于上限"""
        settler = MahjongScoreSettler(is_kou_kou_fan=False)
        # 三家都≥300触发金顶，但部分分数低于上限400
        result = settler._apply_capping([350, 350, 900])
        assert result == [350, 350, 400]

    def test_koukou_golden_cap_mixed(self):
        """口口翻金顶混合值"""
        settler = MahjongScoreSettler(is_kou_kou_fan=True)
        result = settler._apply_capping([320, 960, 320])
        assert result == [320, 500, 320]
