import sys
import os

COMMAND = "python3 tools/visualization.py"
ROOM_NAME = "--room_name scene0191_00"

def visualize_task(task, prediction_path, output_path):
    output_path = os.path.join(output_path, task + '.ply')
    command_str = COMMAND + ' ' + ROOM_NAME + ' ' \
                  + "--prediction_path " + prediction_path + ' ' \
                  + "--task " + task + ' ' \
                  + "--out " + output_path
    print(command_str)
    os.system(command_str)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 visualize.py path_to_result path_to_save_visualized_point_clouds")
        exit()

    visualize_task("input", sys.argv[1], sys.argv[2])
    visualize_task("semantic_gt", sys.argv[1], sys.argv[2])
    visualize_task("semantic_pred", sys.argv[1], sys.argv[2])
    visualize_task("offset_semantic_pred", sys.argv[1], sys.argv[2])
    visualize_task("instance_gt", sys.argv[1], sys.argv[2])
    visualize_task("instance_pred", sys.argv[1], sys.argv[2])

