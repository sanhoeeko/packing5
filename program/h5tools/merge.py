import pandas as pd
import numpy as np


def group_similar_ids(dataframes: list[pd.DataFrame]) -> list[list[str]]:
    try:
        # 处理输入为空的情况
        if not dataframes:
            return []

        # 检查所有DataFrame的列是否一致
        reference_columns = dataframes[0].columns.tolist()
        for df in dataframes:
            if df.columns.tolist() != reference_columns:
                raise ValueError("All DataFrames must have the same columns in the same order.")

        # 检查是否存在'id'列
        if 'id' not in reference_columns:
            raise ValueError("All DataFrames must contain an 'id' column.")

        # 提取非id列
        non_id_columns = [col for col in reference_columns if col != 'id']

        # 处理仅含id列的情况
        if not non_id_columns:
            all_ids = [row['id'] for df in dataframes for _, row in df.iterrows()]
            return [all_ids] if all_ids else []

        # 确定浮点列（基于第一个DataFrame的类型）
        float_cols = [col for col in non_id_columns if pd.api.types.is_float_dtype(dataframes[0][col])]

        # 收集所有行的数据
        entries = []
        for df in dataframes:
            for _, row in df.iterrows():
                entry = {'id': row['id'], 'data': {}}
                for col in non_id_columns:
                    entry['data'][col] = row[col]
                entries.append(entry)

        # 分组处理
        groups = []
        for entry in entries:
            matched = False
            for group in groups:
                # 比较当前entry与组的代表数据
                match = True
                for col in non_id_columns:
                    val = entry['data'][col]
                    rep_val = group['rep_data'][col]

                    # 处理NaN
                    val_nan = pd.isna(val)
                    rep_nan = pd.isna(rep_val)
                    if val_nan and rep_nan:
                        continue
                    if val_nan or rep_nan:
                        match = False
                        break

                    # 浮点列允许误差，其他列严格匹配
                    if col in float_cols:
                        if abs(val - rep_val) > 1e-9:
                            match = False
                            break
                    else:
                        if val != rep_val:
                            match = False
                            break
                if match:
                    group['ids'].append(entry['id'])
                    matched = True
                    break
            if not matched:
                groups.append({'rep_data': entry['data'], 'ids': [entry['id']]})

        # 提取分组结果
        return [group['ids'] for group in groups]

    except Exception as e:
        raise e