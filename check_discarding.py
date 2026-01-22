import sys
sys.path.insert(0, 'D:\\DATA\\Python_Project\\Code\\PettingZooRLENVMahjong')

with open('check_discarding_result.txt', 'w', encoding='utf-8') as f:
    try:
        from src.mahjong_rl.state_machine.states.discarding_state import DiscardingState
        import inspect

        source = inspect.getsource(DiscardingState.step)
        f.write("DiscardingState.step source:\n")
        f.write(source)
        f.write("\n")

        # 检查关键行
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'if action' in line:
                f.write(f"Line {i}: {line}\n")

        f.write("SUCCESS\n")
    except Exception as e:
        f.write(f"ERROR: {e}\n")
        import traceback
        traceback.print_exc(file=f)
