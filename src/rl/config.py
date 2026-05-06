import os
import torch


class RLConfig:
    """RL training configuration with all hyperparameters centralized."""

    def __init__(self, **overrides):
        # --- Model ---
        self.idiom_dim = 384
        self.char_dim = 64
        self.n_heads = 4
        self.n_layers = 3
        self.embedding_type = 'char'
        self.encoder_type = 'cross_attention'

        # --- Environment ---
        self.max_history_len = 64      # attention window (decoupled from game length)
        self.max_game_steps = 200      # hard game truncation limit
        self.max_actions = 600

        # --- PPO ---
        self.lr = 1e-3
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.4
        self.value_coef = 0.5
        self.entropy_coef_start = 0.05
        self.entropy_coef_end = 0.05
        self.ppo_epochs = 1
        self.batch_size = 2048

        # --- Training ---
        self.n_games_per_iter = 512
        self.max_iterations = 15000
        self.eval_interval = 50
        self.eval_games = 800
        self.save_interval = 500

        # --- Curriculum stage limits (0 = skip, large = no effect) ---
        self.stage1_max_iters = 999999
        self.stage2_max_iters = 999999

        # --- Frozen opponent (self-play) ---
        self.use_frozen_opponent = True    # set False for pure same-model self-play
        self.frozen_update_interval = 200  # eval vs frozen every N iters
        self.frozen_win_threshold = 0.55   # update frozen if win rate > this
        self.frozen_eval_games = 400       # games per frozen eval

        # --- Stage 3 opponent mix (pure self-play by default) ---
        self.stage3_self_ratio = 1.0
        self.stage3_random_ratio = 0.0
        self.stage3_qtable_ratio = 0.0

        # --- Paths ---
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        self.data_dir = os.path.join(project_root, 'data')
        self.idiom_file = os.path.join(self.data_dir, 'chinese-xinhua-master',
                                       'data', 'idiom.json')
        if not os.path.exists(self.idiom_file):
            backup = os.path.join(self.data_dir, 'idiom.json')
            if os.path.exists(backup):
                self.idiom_file = backup
        self.results_dir = os.path.join(project_root, 'results')
        self.ckpt_dir = os.path.join(project_root, 'checkpoints')
        self.log_dir = os.path.join(project_root, 'logs')
        self.tensorboard_dir = os.path.join(self.ckpt_dir, 'tensorboard')

        # --- Device ---
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.use_amp = True
            _setup_cuda()
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            self.use_amp = False
        else:
            self.device = 'cpu'
            self.use_amp = False

        # Apply overrides
        for k, v in overrides.items():
            setattr(self, k, v)

    def to_dict(self):
        serializable = (int, float, str, bool, list, dict, tuple, type(None))
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_') and isinstance(v, serializable)}


def _setup_cuda():
    """Enable CUDA optimizations for maximum throughput on RTX 5090."""
    # TF32: faster matmul with negligible precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Auto-tune cuDNN algorithms for fixed-size inputs
    torch.backends.cudnn.benchmark = True
    # Use cuDNN attention if available (PyTorch >= 2.5)
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)


DEFAULT_CONFIG = RLConfig()
