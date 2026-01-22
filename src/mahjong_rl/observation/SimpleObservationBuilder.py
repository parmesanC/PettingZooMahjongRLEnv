from src.mahjong_rl.observation.builder import IObservationBuilder


class SimpleObservationBuilder(IObservationBuilder):
    def build(self, player_id, context):
        return {"player_id": player_id, "hand_tiles": context.players[player_id].hand_tiles}

    def build_action_mask(self, player_id, context):
        return {"draw": True, "discard": True}
