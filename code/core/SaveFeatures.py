import matplotlib as mpl
import datetime
import glob
import pandas as pd
import scipy.stats as stats
from scipy.stats import shapiro, levene, kstest
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import os
import numpy as np
from collections import defaultdict
import pickle
def map_color_dict(key_items,color_map='plasma'):
    '''
    # 提前生成统一的color_dict()
    # color(根据dates元素的数量分配，输出以key_items元素(可以是condition或dates)为key的颜色映射字典)
    '''
    color_vec_norm = np.linspace(0.01,0.99,len(key_items)+1)
    colormap = mpl.colormaps[color_map]
    colors  = colormap(color_vec_norm)
    color_dict = {}
    for i, d in enumerate(key_items):
        color_dict[d] = colors[i]
    return color_dict

def sort_key(filename):
    # 提取条件标识（前两个下划线分割的部分）
    condition = '_'.join(filename.split('_')[:2])
    # 提取编号（最后一个'-'和'.'之间的数字）
    number = int(filename.split('-')[-1].split('.')[0])
    return (priority[condition], number)
#%%
# 定义角度计算公式
def ang_cal(vec_1, vec_2):
    dot_pro = np.dot(vec_1, vec_2)
    mod_1 = np.sqrt(np.dot(vec_1, vec_1))
    mod_2 = np.sqrt(np.dot(vec_2, vec_2))
    if mod_1 == 0 or mod_2 == 0:
        angle = 0
    else:
        cos = dot_pro / (mod_1 * mod_2)
        if np.isnan(cos) == True:
            angle = np.nan
        else:
            angle = np.arccos(round(cos, 1))  #弧度制
    return angle


def clws_delta_phi(vec_1, vec_2, vec_0=[1, 0]):
    # vec_1 is the first vector, and it rotates to the vec_2
    vec_0 = np.array(vec_0)
    agl_1 = ang_cal(vec_1, vec_2)
    agl_2 = ang_cal(vec_0, vec_1)
    agl_3 = ang_cal(vec_0, vec_2)
    if agl_1 == 0:
        agl_1 = 0
        # print('angle = 0°')
    elif agl_1 == np.pi:
        agl_1 = np.pi
        # print('angle = 180°')
    elif vec_1[1] >= 0 and vec_2[1] >= 0:  #同时在第一第二象限
        if (agl_3 - agl_2) > 0:
            agl_1 = (-1) * np.abs(agl_1)
        else:
            agl_1 = np.abs(agl_1)
    elif vec_1[1] <= 0 and vec_2[1] <= 0:  #同时在第三第四象限
        if (agl_3 - agl_2) > 0:
            agl_1 = np.abs(agl_1)
        else:
            agl_1 = (-1) * np.abs(agl_1)
    elif vec_1[1] >= 0 and vec_2[1] <= 0:
        if (agl_3 + agl_2) > np.pi:
            agl_1 = (-1) * np.abs(agl_1)
        else:  #如果等于180°认为是顺时针旋转
            agl_1 = np.abs(agl_1)
    elif vec_1[1] <= 0 and vec_2[1] >= 0:
        if (agl_3 + agl_2) < np.pi:
            agl_1 = (-1) * np.abs(agl_1)
        else:  #如果等于180°认为是顺时针旋转
            agl_1 = np.abs(agl_1)
    return agl_1 * 180 / np.pi


def realign_xy(x, y, sin_phi, cos_phi):
    """
    旋转坐标 (x, y) 角度，由 sin_phi 和 cos_phi 定义。
    """
    rotation_mat = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
    return np.matmul(rotation_mat, np.array([x, y]))


def realign_coordinate(df, df_roi):
    """
    旋转坐标轴，将X轴对齐到两琼脂块中心的连线。
    """
    # 提取points并调整原点到左下角点
    points = np.array([list(map(int, p.strip('()').split(','))) for p in df_roi.iloc[0, :-1]])
    origin = points[1]
    points -= origin
    df_realign = df.copy()
    df_realign[['X', 'Y']] -= origin

    # 计算旋转角度
    new_x_axis = points[2] - points[1]
    mod = np.linalg.norm(new_x_axis)
    sin_phi, cos_phi = new_x_axis[1] / mod, new_x_axis[0] / mod

    # 旋转df
    df_realign[['X', 'Y']] = df_realign.apply(
        lambda row: realign_xy(row.X, row.Y, sin_phi, cos_phi), axis=1, result_type='expand'
    )

    # 旋转points
    rotated_points = np.array([realign_xy(p[0], p[1], sin_phi, cos_phi) for p in points])

    # 恢复原点
    df_realign[['X', 'Y']] += origin
    rotated_points += origin

    # 返回结果
    point_fin = {f'point{i + 1}': [rotated_points[i]] for i in range(4)}
    df_ROI_fin = pd.DataFrame(point_fin)
    return df_realign, df_ROI_fin


def extract_single_dataframe_parameters(df, excel_file, time_len=1.5 * 3600):
    # 提取 ROI 顶点
    roi_file = (os.path.basename(excel_file)).replace('_second_clear', '_rectangle_info')
    roi_path = os.path.join(os.path.dirname(excel_file), roi_file)
    df_ROI = pd.read_csv(roi_path)
    points = [list(map(int, p.strip('()').split(','))) for p in df_ROI.iloc[0, :-1]]

    # 去噪并筛选虫子轨迹
    noise_lim = 250
    df_worms = df[df.Diagonal_Length > noise_lim]
    print(f'Noise limit: {noise_lim}, Worm trajectories: {len(df_worms.ID.unique())} IDs')

    # 坐标轴重定位
    df_worms_aligned, df_ROI_aligned = realign_coordinate(df_worms, df_ROI)
    df = df_worms_aligned
    unique_ids = df['ID'].unique()
    frame_rate = max(df.Timestamp) / time_len
    print('frame rate', frame_rate)
    # 滑动窗口特征计算
    processed_data = []
    for worm_id in unique_ids:
        print(worm_id)
        try:
            window_size = 1
            # 滤波与预处理
            df_slide = df[df.ID == worm_id].copy()
            df_slide['X'] = df_slide['X'].rolling(window=window_size, center=True).mean()
            #把原点调到左上角
            df_slide['Y'] = (HEIGHT - df_slide['Y']).rolling(window=window_size, center=True).mean()
            # 数据准备
            trajectory = df_slide[['X', 'Y', 'Timestamp']].values.astype(float)
            x, y, timestamps = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2] / frame_rate

            # 定义辅助函数
            def compute_delta(data, bins, scale_factor=1):
                return (data[bins:] - data[:-bins]) * scale_factor

            def smooth_velocity(x, y, timestamps, bins, pixel_length):
                delta_x = compute_delta(x, bins, pixel_length)
                delta_y = compute_delta(y, bins, pixel_length)
                delta_t = compute_delta(timestamps, bins)
                velocity = np.column_stack((delta_x / delta_t, delta_y / delta_t))
                speed = np.linalg.norm(velocity, axis=1)
                return velocity, speed, delta_t

            # 速度计算与异常检测
            pixel_length = plate_width / WIDTH

            bins_spd = 2
            velocity_spd, speed_spd, delta_t_spd = smooth_velocity(x, y, timestamps, bins_spd, pixel_length)
            # 补充 NaN 填充
            velocity_spd = np.pad(velocity_spd, ((bins_spd // 2, bins_spd // 2), (0, 0)), constant_values=np.nan)
            speed_spd = np.pad(speed_spd, (bins_spd // 2, bins_spd // 2), constant_values=np.nan)
            # 角速度计算
            hlf_spd_agl_inv = 0.1
            bins_agl = 20
            hlf_agl_inv = 0.1
            hlf_bins_agl = 10
            velocity_agl, _, time_step_agl = smooth_velocity(x, y, timestamps, bins_agl, pixel_length)

            angular_velocity = np.zeros(len(velocity_agl))
            valid_idx = np.arange(hlf_bins_agl, len(time_step_agl) - hlf_bins_agl)
            for i in valid_idx:
                delta_phi = clws_delta_phi(velocity_agl[i - hlf_bins_agl], velocity_agl[i + hlf_bins_agl])
                delta_t = time_step_agl[i] * hlf_bins_agl
                angular_velocity[i] = delta_phi / delta_t
            angular_velocity = np.pad(angular_velocity, (hlf_bins_agl, hlf_bins_agl), constant_values=np.nan)
            # CTX计算
            vec_left, vec_right = np.array([-1, 0]), np.array([1, 0])
            mod_vec_left, mod_vec_right = np.linalg.norm(vec_left), np.linalg.norm(vec_right)
            mod_velocity_spd = np.linalg.norm(velocity_spd, axis=1)
            dot_products = np.dot(velocity_spd, vec_left)
            ctx = np.divide(dot_products, mod_velocity_spd * mod_vec_left, out=np.zeros_like(dot_products),
                            where=(mod_velocity_spd * mod_vec_left != 0))
            # True Displacement to middle line
            disp_to_mid_distribution = (x - (WIDTH + 2 * bin) / 2) * pixel_length * 10
            disp_to_mid_errorbar = ((WIDTH + 2 * bin) / 2 - x) * pixel_length * 10
            # 合并结果
            df_slide['speed'] = speed_spd
            df_slide['x_velocity'], df_slide['y_velocity'] = velocity_spd[:, 0], velocity_spd[:, 1]
            df_slide['agl_velocity'] = angular_velocity
            df_slide['agl_speed'] = np.abs(angular_velocity)
            df_slide['CTX'] = ctx
            df_slide['X_org'] = df[df.ID == worm_id]['X']
            df_slide['Y_org'] = df[df.ID == worm_id]['Y']
            df_slide['Disp_to_mid_mm_dis'] = disp_to_mid_distribution
            df_slide['Disp_to_mid_mm_err'] = disp_to_mid_errorbar
            processed_data.append(df_slide)
        except Exception as e:
            print(f'Error processing ID {worm_id}: {e}')
    df_all = pd.concat(processed_data)
    return processed_data, df_all


def MakeSubFolder(path, subfolder_name):
    new_path = os.path.join(path, subfolder_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path

WIDTH = 4100
HEIGHT = 3500
plate_width = 10
bin = 10
time_len = 1.5*3600
#顺浓度梯度或逆浓度梯度
vec_left, vec_right = [-1,0], [1,0]
# 自定义排序优先级
priority = {
    '0g_control': 0,
    '0g_end': 1,
    '4.5g_end': 2,
    '4.5g_control': 3
}
PREDEFINED_ORDER = [
    "0g_control",
    "0g_end",
    "4.5g_end",
    "4.5g_control"
]
PREDEFINEDblankORDER = [
    "3g-1gNa-1h",
    "3g-1gNa-4h",
    "1gNa-end(8x)",
    "5gNa-end(8x)",
    "3g-5gNa-4h",
    "3g-5gNa-1h"
]
COLOR_MAP = map_color_dict(PREDEFINEDblankORDER, color_map='plasma')
# 读取目标文件
read_mian_path = r'E:\ZHY\code\AssociativeModel\cal\experimental data'
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
save_main_path = os.path.join(r"E:\ZHY\code\AssociativeModel\Fitting", current_date)
if not os.path.exists(save_main_path):
    os.makedirs(save_main_path)
condition_dict = defaultdict(list)
global_max_time = 0
# target_dirs = ['FitTest']
target_dirs = ['1g-1h', '1g-4h', '1g-end', '5g-1h', '5g-4h', '5g-end']
condition_parser = lambda x: (x.split('_')[3])
date_parser = lambda x: (x.split('_')[0])
for target_dir in target_dirs:
   dir_path = os.path.join(read_mian_path, target_dir)
   if os.path.exists(dir_path):
       file_pattern = os.path.join(dir_path,"**","*second_clear.csv")
       csv_files = glob.glob(file_pattern, recursive=True)
       for file_path in csv_files:
           df = pd.read_csv(file_path)
           base_name = os.path.splitext(os.path.basename(file_path))[0]
           condition = condition_parser(base_name)
           date = date_parser(base_name)
           _, df = extract_single_dataframe_parameters(df, file_path, time_len=1.5*3600)
           current_max = df['Timestamp'].max()
           global_max_time = max(global_max_time, current_max)
           condition_dict[condition].append(df)
with open('condition_dict.pkl', 'wb') as f:
    pickle.dump(condition_dict, f)