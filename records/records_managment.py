import os

from lib.report_utils import save_new_data_point

base_directory=r'records/'

def try_remove(file):
    try:
        os.remove(file)
    except Exception as e:
        pass
def clear_records():
    try_remove('generated_pose_std.txt')
    try_remove('generated_pose_ave.txt')
    try_remove('generated_pose_max.txt')
    try_remove('generated_pose_min.txt')
    try_remove('generated_pose_cv.txt')

    try_remove('discriminator_loss.txt')
    try_remove('curriculum_loss.txt')
    try_remove('generator_loss.txt')
    try_remove('collision_times.txt')
    try_remove('collision__free_times.txt')
    try_remove('good_approach_times.txt')

    try_remove('quality_score.txt')

def save_record(data,file):
    save_new_data_point(data, base_directory+file)
