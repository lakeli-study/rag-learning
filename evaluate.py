#事先说明，本段评估程序是为了评估大模型在不同场景下的回答质量，而不是为了评估大模型的性能指标，
#评估我这一文件夹内的测试代码的，可以放在某个RAG串联路下方。

import json


# ---------------------- 1. 待测试大模型调用函数（模拟） ----------------------
def llm_invoke(prompt):
    # 这里可以替换成常见的大模型接口，比如我常用的glm-4-flash智普模型
    return "模型输出结果"

# ---------------------- 2. 第一轮Eval，收集失败案例 ----------------------
def run_eval(test_cases, save_fail_path="fail_cases.json"):
    fail_list = []
    for case in test_cases:
        input_text = case["input"]
        expect = case["expect"]
        output = llm_invoke(input_text)

        # 你的判定逻辑：判断输出是否符合预期（可以是关键词匹配、embedding相似度）
        is_pass = judge_result(output, expect)
        if not is_pass:
            # 记录失败样本：输入、预期、实际输出
            fail_case = {
                "input": input_text,
                "expect": expect,
                "actual_output": output
            }
            fail_list.append(fail_case)
            print(f"❌ 失败：{input_text}")
    
    # 持久化保存失败案例，留作回归测试
    with open(save_fail_path, "w", encoding="utf-8") as f:
        json.dump(fail_list, f, ensure_ascii=False, indent=2)
    print(f"\n评估结束，共 {len(fail_list)} 条失败案例，已存入 {save_fail_path}")
    return fail_list

# 结果判定函数，按需改写
def judge_result(actual, expect):
    # 简单示例：预期关键词是否出现在回答里
    return expect in actual


# ---------------------- 3. 回归测试：只跑历史失败案例 ----------------------
def run_regression_test(fail_case_path="fail_cases.json"):
    # 读取上一轮eval存下来的失败用例
    with open(fail_case_path, "r", encoding="utf-8") as f:
        fail_cases = json.load(f)
    
    new_fail = []
    fixed_count = 0
    for case in fail_cases:
        output = llm_invoke(case["input"])
        is_pass = judge_result(output, case["expect"])
        if is_pass:
            fixed_count += 1
            print(f"✅ 已修复：{case['input']}")
        else:
            new_fail.append(case)
            print(f"❌ 依旧失败：{case['input']}")
    
    print(f"\n回归测试汇总：原有 {len(fail_cases)} 个问题，本次修复 {fixed_count} 个，剩余未修复 {len(new_fail)}")
    # 可选：覆盖更新失败案例库，只保留还没修好的case
    with open(fail_case_path, "w", encoding="utf-8") as f:
        json.dump(new_fail, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 原始测试集
    test_set = [
        {"input": "1+1等于几", "expect": "2"},
        {"input": "2*3等于几", "expect": "6"}
    ]

    # 第一步：完整评估，产出失败案例
    # import json

# ---------------------- 1. 待测试大模型调用函数（模拟） ----------------------
def llm_invoke(prompt):
    # 这里替换成你真实的LLM接口调用
    return "模型输出结果"

# ---------------------- 2. 第一轮Eval，收集失败案例 ----------------------
def run_eval(test_cases, save_fail_path="fail_cases.json"):
    fail_list = []
    for case in test_cases:
        input_text = case["input"]
        expect = case["expect"]
        output = llm_invoke(input_text)

        # 你的判定逻辑：判断输出是否符合预期（可以是关键词匹配、embedding相似度）
        is_pass = judge_result(output, expect)
        if not is_pass:
            # 记录失败样本：输入、预期、实际输出
            fail_case = {
                "input": input_text,
                "expect": expect,
                "actual_output": output
            }
            fail_list.append(fail_case)
            print(f"❌ 失败：{input_text}")
    
    # 持久化保存失败案例，留作回归测试
    with open(save_fail_path, "w", encoding="utf-8") as f:
        json.dump(fail_list, f, ensure_ascii=False, indent=2)
    print(f"\n评估结束，共 {len(fail_list)} 条失败案例，已存入 {save_fail_path}")
    return fail_list

# 结果判定函数，按需改写
def judge_result(actual, expect):
    # 简单示例：预期关键词是否出现在回答里
    return expect in actual


# ---------------------- 3. 回归测试：只跑历史失败案例 ----------------------
def run_regression_test(fail_case_path="fail_cases.json"):
    # 读取上一轮eval存下来的失败用例
    with open(fail_case_path, "r", encoding="utf-8") as f:
        fail_cases = json.load(f)
    
    new_fail = []
    fixed_count = 0
    for case in fail_cases:
        output = llm_invoke(case["input"])
        is_pass = judge_result(output, case["expect"])
        if is_pass:
            fixed_count += 1
            print(f"✅ 已修复：{case['input']}")
        else:
            new_fail.append(case)
            print(f"❌ 依旧失败：{case['input']}")
    
    print(f"\n回归测试汇总：原有 {len(fail_cases)} 个问题，本次修复 {fixed_count} 个，剩余未修复 {len(new_fail)}")
    # 可选：覆盖更新失败案例库，只保留还没修好的case
    with open(fail_case_path, "w", encoding="utf-8") as f:
        json.dump(new_fail, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 原始测试集
    test_set = [
        {"input": "1+1等于几", "expect": "2"},
        {"input": "2*3等于几", "expect": "6"}
    ]

    # 第一步：完整评估，产出失败案例
    # run_eval(test_set)

    run_eval(test_set)


    # 修改prompt/模型版本之后，第二步：跑回归，校验旧问题
    # run_regression_test()
    run_regression_test()   