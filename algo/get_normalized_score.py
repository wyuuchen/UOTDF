REF_SCORES = {
    "halfcheetah": {
        "gravity": (-280.18, 9509.15),
        "kinematic": (-280.18, 7065.03),
        "morphology": (-280.18, 9713.59)
    },
    "hopper": {
        "gravity": (-26.336, 3234.3),
        "kinematic": (-26.336, 2842.73),
        "morphology": (-26.336, 3152.75)
    },
    "walker2d": {
        "gravity": (10.08, 5194.713),
        "kinematic": (10.08, 3257.51),
        "morphology": (10.08, 4398.43)
    },
    "ant": {
        "gravity": (-325.6, 4317.065),
        "kinematic": (-325.6, 5122.57),
        "morphology": (-325.6, 5722.01)
    }
}

def get_normalized_score(env_name_str, raw_score):
    """
    自动从 env 字符串中解析环境名和 Shift 类型，并计算归一化分数。
    Example env_name_str: "halfcheetah-gravity", "ant_kinematic_medium"
    """
    s = env_name_str.lower()
    
    # 1. 解析 Base Environment Name
    base_env = None
    if 'halfcheetah' in s:
        base_env = 'halfcheetah'
    elif 'hopper' in s:
        base_env = 'hopper'
    elif 'walker' in s: # 涵盖 walker 和 walker2d
        base_env = 'walker2d'
    elif 'ant' in s:
        base_env = 'ant'
        
    # 2. 解析 Shift Type
    shift_type = None
    if 'gravity' in s:
        shift_type = 'gravity'
    elif 'kinematic' in s:
        shift_type = 'kinematic'
    elif 'morph' in s: # 涵盖 morphology 和 morph
        shift_type = 'morphology'
        
    # 3. 安全检查与计算
    if base_env is None or shift_type is None:
        # 如果解析失败（比如是标准环境），返回原始分数或尝试使用 D4RL 默认归一化
        return raw_score 
        
    if shift_type not in REF_SCORES[base_env]:
        print(f"[Warning] No reference score for {base_env}-{shift_type}. Return raw score.")
        return raw_score

    min_score, max_score = REF_SCORES[base_env][shift_type]
    
    # 归一化公式: 100 * (score - min) / (max - min)
    norm_score = 100 * (raw_score - min_score) / (max_score - min_score)
    return norm_score