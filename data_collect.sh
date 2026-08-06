# bash evaluations/run_eval.sh robocasa_multi_task \
#   'env.eval.task_names=[CloseDrawer]' \
#   rollout.safe_dump.enabled=true \
#   rollout.safe_dump.save_dir=/mnt/public2/lizuojun/Codespace/RLinf/data/safe_rollouts/CloseDrawer
#
# bash evaluations/run_eval.sh robocasa_multi_task \
#   'env.eval.task_names=[OpenSingleDoor]' \
#   rollout.safe_dump.enabled=true \
#   rollout.safe_dump.save_dir=/mnt/public2/lizuojun/Codespace/RLinf/data/safe_rollouts/OpenSingleDoor
#
# bash evaluations/run_eval.sh robocasa_multi_task \
#   'env.eval.task_names=[TurnOnSinkFaucet]' \
#   rollout.safe_dump.enabled=true \
#   rollout.safe_dump.save_dir=/mnt/public2/lizuojun/Codespace/RLinf/data/safe_rollouts/TurnOnSinkFaucet

# bash evaluations/run_eval.sh robocasa_multi_task \
#   'env.eval.task_names=[CloseDrawer,OpenSingleDoor,TurnOnSinkFaucet]' \
#   rollout.safe_dump.enabled=true \
#   rollout.safe_dump.save_dir=/mnt/public2/lizuojun/Codespace/RLinf/data/safe_rollouts/Test

# bash evaluations/run_eval.sh robocasa_multi_task \
#   'env.eval.task_names=[CloseDrawer]'

# bash evaluations/run_eval.sh robocasa_multi_task \
#   'env.eval.task_names=[OpenSingleDoor]'

# bash evaluations/run_eval.sh robocasa_multi_task \
#   'env.eval.task_names=[TurnOnSinkFaucet]'


bash evaluations/run_eval.sh libero_10_openpi_pi0_eval \
    'env.eval.task_id_filter=[0]'

bash evaluations/run_eval.sh libero_10_openpi_pi0_eval \
    'env.eval.task_id_filter=[1]'

bash evaluations/run_eval.sh libero_10_openpi_pi0_eval \
    'env.eval.task_id_filter=[2]'

bash evaluations/run_eval.sh libero_10_openpi_pi0_eval \
    'env.eval.task_id_filter=[3]'

bash evaluations/run_eval.sh libero_10_openpi_pi0_eval \
    'env.eval.task_id_filter=[4]'

bash evaluations/run_eval.sh libero_10_openpi_pi0_eval \
    'env.eval.task_id_filter=[5]'

bash evaluations/run_eval.sh libero_10_openpi_pi0_eval \
    'env.eval.task_id_filter=[6]'

bash evaluations/run_eval.sh libero_10_openpi_pi0_eval \
    'env.eval.task_id_filter=[7]'

bash evaluations/run_eval.sh libero_10_openpi_pi0_eval \
    'env.eval.task_id_filter=[8]'

bash evaluations/run_eval.sh libero_10_openpi_pi0_eval \
    'env.eval.task_id_filter=[9]'

