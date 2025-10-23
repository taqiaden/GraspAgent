from lib.Multible_planes_detection.plane_detecttion import bin_planes_detection


def analytical_bin_mask(pc, file_ids):
    try:
        bin_mask,floor_elevation = bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015, view=False,
                                        file_index=file_ids[0], cache_name='bin_planes2')
    except Exception as error_message:
        print(file_ids[0])
        print(error_message)
        bin_mask = None
    return bin_mask