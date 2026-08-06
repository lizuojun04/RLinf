import os

data_path = './data/libero/'

for task_id_path in os.listdir(data_path):
    task_id = task_id_path.split('.')[0]
    success_count = 0
    fail_count = 0
    # print('-' * 40)
    for dir in os.listdir(os.path.join(data_path, task_id_path)):
        # print(dir)
        for rollout_dir in os.listdir(os.path.join(data_path, task_id_path, dir)):
            if rollout_dir == 'success':
                success_count += len(os.listdir(os.path.join(data_path, task_id_path, dir, rollout_dir)))
            elif rollout_dir== 'fail':
                fail_count += len(os.listdir(os.path.join(data_path, task_id_path, dir, rollout_dir)))
        # print('+' * 10)
    print(f'Task ID: {task_id}, Success Count: {success_count}, Fail Count: {fail_count}')

