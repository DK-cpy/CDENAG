import warnings
import sys
import re
from datetime import datetime
from collections import OrderedDict
from TransNASBench101.api import TransNASBenchAPI as API
from experiments import main_exp
from config.config import hyper_params_setting

# 生成带时间戳的结果文件名
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULT_FILE = f"experiment_results_{timestamp_str}.txt"

# 任务缩写映射（键为 API 返回的任务名）
TASK_ABBREV = {
    "class_object": "OC",
    "class_scene": "SC",
    "autoencoder": "AE",
    "normal": "SE",               # Surface Normal Estimation
    "segmentsemantic": "SS",      # Semantic Segmentation
    "room_layout": "RP",
    "jigsaw": "JS"
}

class SaveLogger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.file = open(file_path, "w", encoding="utf-8")
        self.file.write("=" * 80 + "\n")
        self.file.write(f"TransNAS-Bench-101 实验记录 | 开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.write("=" * 80 + "\n\n")
        self.file.flush()
        # 用于存储解析到的结果行
        self.results = []   # 每个元素为 (task, space, seed, accuracy)

    def write(self, message):
        self.terminal.write(message)
        # 所有内容都写入文件（如果想精简可保留原有过滤）
        self.file.write(message)
        self.file.flush()

        # 捕获形如 ">>> Task class_object with seed 1 in search space macro, max accuracy: 47.96 %" 的行
        # 采用正则匹配
        match = re.search(
            r">>> Task (\S+?) with seed (\d+) in search space (\S+?), max accuracy: ([\d\.]+) %",
            message
        )
        if match:
            task = match.group(1)
            seed = int(match.group(2))
            space = match.group(3)
            acc = float(match.group(4))
            self.results.append((task, space, seed, acc))

    def flush(self):
        pass

    def close(self):
        self.file.close()
        sys.stdout = self.terminal


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logger = SaveLogger(RESULT_FILE)
    sys.stdout = logger

    # 加载NAS基准工具（experiments.py 内部也会加载，但此处仅为了获取任务列表）
    path2nas_bench_file = "TransNASBench101/transnas-bench_v10141024.pth"
    api = API(path2nas_bench_file)
    task_list = api.task_list           # 保持原始任务名
    search_spaces = ["macro", "micro"]

    # ============== 依次运行所有任务 ==============
    for space in search_spaces:
        print(f"\n{'='*55}")
        print(f" 开始运行 【{space.upper()}】 搜索空间 - 7个任务 ")
        print(f"{'='*55}\n")

        for task in task_list:
            try:
                print(f"\n🎯 启动任务：{space} | {task}")
                # main_exp 内部会遍历该任务的所有种子并打印结果
                main_exp(task=task, search_space=space)
                print(f"\n✅ {space.upper()} - {task} 运行完成！\n")
            except Exception as e:
                print(f"\n❌ {space.upper()} - {task} 运行失败：{str(e)}\n")
                continue

    # ========== 统一汇总输出 ==========
    sys.stdout = logger.terminal   # 暂时切回终端，以便汇总只显示在控制台
    summary = []
    summary.append("\n" + "=" * 80)
    summary.append("                    最终结果汇总")
    summary.append("=" * 80)

    # logger.results 中已经收集了所有任务所有种子的结果
    for (task, space, seed, acc) in logger.results:
        abbrev = TASK_ABBREV.get(task, task.upper())
        line = (f">>> Task {task} ({abbrev}) with seed {seed} "
                f"in search space {space}, max accuracy: {acc:.2f} %")
        summary.append(line)
    summary.append("=" * 80 + "\n")

    for line in summary:
        print(line)

    # 同时追加写入日志文件
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        for line in summary:
            f.write(line + "\n")

    print(f"\n🎉 全部任务执行完毕！结果已保存至：{RESULT_FILE}")
    logger.close()

'''
import warnings
import sys
import time  # 【新增】导入时间模块
from datetime import datetime
from TransNASBench101.api import TransNASBenchAPI as API
from experiments import main_exp
# 正确导入你的超参数配置（无报错版）
from config.config import hyper_params_setting

# ====================== 自动生成带时间戳的文件名 ======================
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULT_FILE = f"experiment_results_{timestamp_str}.txt"
# ======================================================================

class SaveLogger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.file = open(file_path, "w", encoding="utf-8")
        self.file.write("="*80 + "\n")
        self.file.write(f"TransNAS-Bench-101 实验记录 | 开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.write("="*80 + "\n\n")
        self.file.flush()

    def write(self, message):
        self.terminal.write(message)
        
        # 【新增】捕获关键词：加入 "Task completed" 和 "Total time:"
        capture_keys = [
            "[EvoDiff] Starting evolution process",
            "Task:",
            "Search Space:",
            "Total steps:",
            "Population size:",
            "Using ConsistencyGenerator:",
            "Single-step mode:",
            "[ConsistencyGenerator Params]",
            "Teacher model:",
            "Lambda KL:",
            "Lambda CE:",
            "Lambda Consistency:",
            "Consistency type:",
            "Perturb scale:",
            "Tournament size:",
            "History max len:",
            "开始运行",
            "启动任务",
            "max accuracy",
            # --- 新增用于捕获结束信息的关键词 ---
            "[EvoDiff] Task completed",
            "Total time:",
            "运行完成",
            "运行失败",
            "全部任务执行完毕"
        ]
        
        if any(key in message for key in capture_keys):
            self.file.write(message.strip() + "\n")
            self.file.flush()

    def flush(self):
        pass

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sys.stdout = SaveLogger(RESULT_FILE)

    # 加载NAS基准工具
    path2nas_bench_file = "TransNASBench101/transnas-bench_v10141024.pth"
    api = API(path2nas_bench_file)
    task_list = api.task_list
    search_spaces = ["macro", "micro"]

    # ============== 自动运行所有任务 ==============
    for space in search_spaces:
        print(f"\n{'='*55}")
        print(f" 开始运行 【{space.upper()}】 搜索空间 - 7个任务 ")
        print(f"{'='*55}\n")
        
        for task in task_list:
            try:
                seed = hyper_params_setting[space][task]['seed'][0]
                print(f"\n🎯 启动任务：{space} | {task} | 种子：{seed}")
                
                # ====================== 【核心修改】加入计时逻辑 ======================
                task_start_time = time.time()
                
                # 运行核心实验
                main_exp(task=task, search_space=space)
                
                # 计算耗时
                task_duration = time.time() - task_start_time
                
                # 模仿 evo_diff 风格打印结束信息
                print(f"\n{'='*60}")
                print(f"[EvoDiff] Task completed (TransNAS-Bench-101)")
                print(f"  - Task:         {task}")
                print(f"  - Search Space: {space}")
                print(f"  - Total time:   {task_duration:.2f}s")
                print(f"{'='*60}")
                # =======================================================================
                
                print(f"\n✅ {space.upper()} - {task} 运行完成！\n")
                
            except Exception as e:
                print(f"\n❌ {space.upper()} - {task} 运行失败：{str(e)}\n")
                continue

    print(f"\n🎉 全部任务执行完毕！结果已保存至：{RESULT_FILE}")
    '''