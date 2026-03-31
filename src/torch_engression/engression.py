import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .loss import energy_loss, energy_loss_two_sample
from .models import StoNet
from .data.loader import make_dataloader
from .utils import auto_device, vectorize, cor, make_folder, set_seed


def engression(x, y, classification=False,
               num_layer=2, hidden_dim=100, noise_dim=100, out_act=None,
               add_bn=True, resblock=False, beta=1,
               lr=0.0001, num_epochs=500, batch_size=None,
               device=None, standardize=True, verbose=True,
               compile_model=False, seed=None):
    """Fit an engression model (convenience function).

    Args:
        x (torch.Tensor): training predictors (n, d_x).
        y (torch.Tensor): training responses (n, d_y).
        classification (bool): classification mode. Default False.
        num_layer (int): number of layers. Default 2.
        hidden_dim (int): neurons per layer. Default 100.
        noise_dim (int): noise dimension. Default 100.
        out_act (str): output activation. Default None.
        add_bn (bool): batch normalization. Default True.
        resblock (bool): residual blocks. Default False.
        beta (float): energy loss power parameter. Default 1.
        lr (float): learning rate. Default 0.0001.
        num_epochs (int): training epochs. Default 500.
        batch_size (int): batch size. Default None (full batch).
        device: "cpu", "cuda", "gpu", None (auto-detect). Default None.
        standardize (bool): standardize data internally. Default True.
        verbose (bool): show progress. Default True.
        compile_model (bool): apply torch.compile to StoNet. Default False.
        seed (int): random seed for reproducibility. Default None.

    Returns:
        Engressor: fitted model.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples.")
    engressor = Engressor(
        in_dim=x.shape[1], out_dim=y.shape[1], classification=classification,
        num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim,
        out_act=out_act, resblock=resblock, add_bn=add_bn, beta=beta,
        lr=lr, num_epochs=num_epochs, batch_size=batch_size,
        standardize=standardize, device=device, verbose=verbose,
        compile_model=compile_model, seed=seed)
    engressor.train(x, y, num_epochs=num_epochs, batch_size=batch_size,
                    standardize=standardize, verbose=verbose)
    return engressor


class Engressor:
    """GPU-accelerated distributional regression via energy scores.

    Wraps a stochastic neural network (StoNet) trained with the energy score
    loss to learn the full conditional distribution P(Y|X).

    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        classification (bool): classification mode. Default False.
        num_layer (int): number of layers. Default 2.
        hidden_dim (int): neurons per layer. Default 100.
        noise_dim (int): noise dimension. Default 100.
        out_act: output activation. Default None.
        resblock (bool): residual blocks. Default False.
        add_bn (bool): batch normalization. Default True.
        beta (float): energy loss power. Default 1.
        lr (float): learning rate. Default 0.0001.
        num_epochs (int): training epochs. Default 500.
        batch_size (int): batch size. Default None (full batch).
        standardize (bool): standardize data. Default True.
        device: device. Default None (auto-detect).
        compile_model (bool): torch.compile the StoNet. Default False.
        seed (int): random seed. Default None.
    """

    def __init__(self, in_dim, out_dim, classification=False,
                 num_layer=2, hidden_dim=100, noise_dim=100,
                 out_act=None, resblock=False, add_bn=True, beta=1,
                 lr=0.0001, num_epochs=500, batch_size=None, standardize=True,
                 device=None, verbose=True, compile_model=False, seed=None):
        super().__init__()
        if seed is not None:
            set_seed(seed)

        self.classification = classification
        if classification:
            out_act = "softmax"
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.out_act = out_act
        self.resblock = resblock
        self.add_bn = add_bn
        self.beta = beta
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.standardize = standardize
        self.device = auto_device(device)
        self.compile_model = compile_model

        if verbose:
            print(f"Using device: {self.device}")

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        self.model = StoNet(
            in_dim, out_dim, num_layer, hidden_dim, noise_dim,
            add_bn, out_act, resblock, compile_model=compile_model,
            verbose=verbose).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.verbose = verbose
        self.tr_loss = None

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def summary(self):
        """Print model architecture and training loss."""
        print(f"Engression model\n"
              f"  layers: {self.num_layer}\n"
              f"  hidden_dim: {self.hidden_dim}\n"
              f"  noise_dim: {self.noise_dim}\n"
              f"  resblock: {self.resblock}\n"
              f"  epochs: {self.num_epochs}\n"
              f"  batch_size: {self.batch_size}\n"
              f"  lr: {self.lr}\n"
              f"  standardize: {self.standardize}\n"
              f"  device: {self.device}\n"
              f"  training: {self.model.training}")
        if self.tr_loss is not None:
            print(f"  energy_loss: {self.tr_loss[0]:.4f}\n"
                  f"  E|Y-Yhat|: {self.tr_loss[1]:.4f}\n"
                  f"  E|Yhat-Yhat'|: {self.tr_loss[2]:.4f}")

    def _standardize_data_and_record_stats(self, x, y):
        """Standardize and record stats for later unstandardization."""
        self.x_mean = torch.mean(x, dim=0)
        self.x_std = torch.std(x, dim=0)
        self.x_std[self.x_std == 0] += 1e-5
        if not self.classification:
            self.y_mean = torch.mean(y, dim=0)
            self.y_std = torch.std(y, dim=0)
            self.y_std[self.y_std == 0] += 1e-5
        else:
            self.y_mean = torch.zeros(y.shape[1:], device=y.device).unsqueeze(0)
            self.y_std = torch.ones(y.shape[1:], device=y.device).unsqueeze(0)
        x_s = (x - self.x_mean) / self.x_std
        y_s = (y - self.y_mean) / self.y_std
        self.x_mean = self.x_mean.to(self.device)
        self.x_std = self.x_std.to(self.device)
        self.y_mean = self.y_mean.to(self.device)
        self.y_std = self.y_std.to(self.device)
        return x_s, y_s

    def standardize_data(self, x, y=None):
        """Standardize using recorded stats."""
        if not self.standardize:
            return (x, y) if y is not None else x
        if y is None:
            return (x - self.x_mean) / self.x_std
        return (x - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std

    def unstandardize_data(self, y, x=None, expand_dim=False):
        """Transform predictions back to original scale."""
        if not self.standardize:
            return (x, y) if x is not None else y
        if x is None:
            if expand_dim:
                return y * self.y_std.unsqueeze(0).unsqueeze(2) + self.y_mean.unsqueeze(0).unsqueeze(2)
            return y * self.y_std + self.y_mean
        return x * self.x_std + self.x_mean, y * self.y_std + self.y_mean

    def train(self, x, y, num_epochs=None, batch_size=None, lr=None,
              standardize=None, verbose=True):
        """Train the model with optional AMP on CUDA.

        Args:
            x (torch.Tensor): training predictors.
            y (torch.Tensor): training responses.
            num_epochs (int): override epochs.
            batch_size (int): override batch size.
            lr (float): override learning rate.
            standardize (bool): override standardize.
            verbose (bool): show progress bar.
        """
        self.train_mode()
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is None:
            batch_size = self.batch_size if self.batch_size is not None else x.size(0)
        if lr is not None and lr != self.lr:
            self.lr = lr
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if standardize is not None:
            self.standardize = standardize

        x = vectorize(x)
        y = vectorize(y)
        if self.standardize:
            x, y = self._standardize_data_and_record_stats(x, y)
        x = x.to(self.device)
        y = y.to(self.device)

        # AMP setup (CUDA only)
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        if batch_size >= x.size(0) // 2:
            # Full-batch training
            self.batch_size = x.size(0)
            pbar = tqdm(range(self.num_epochs), desc="Training",
                        disable=not verbose, leave=True)
            for epoch_idx in pbar:
                self.model.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    y_sample1 = self.model(x)
                    y_sample2 = self.model(x)
                # Loss in FP32 to avoid catastrophic cancellation
                loss, loss1, loss2 = energy_loss_two_sample(
                    y.float(), y_sample1.float(), y_sample2.float(),
                    beta=self.beta, verbose=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "E|Y-Ŷ|": f"{loss1.item():.4f}",
                    "E|Ŷ-Ŷ'|": f"{loss2.item():.4f}",
                })
        else:
            # Mini-batch training
            pin = self.device.type == "cuda"
            train_loader = make_dataloader(x, y, batch_size=batch_size,
                                           shuffle=True, pin_memory=pin)
            pbar = tqdm(range(self.num_epochs), desc="Training",
                        disable=not verbose, leave=True)
            for epoch_idx in pbar:
                epoch_loss = 0.0
                epoch_loss1 = 0.0
                epoch_loss2 = 0.0
                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    self.model.zero_grad()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        y_s1 = self.model(x_batch)
                        y_s2 = self.model(x_batch)
                    loss, loss1, loss2 = energy_loss_two_sample(
                        y_batch.float(), y_s1.float(), y_s2.float(),
                        beta=self.beta, verbose=True)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    epoch_loss += loss.item()
                    epoch_loss1 += loss1.item()
                    epoch_loss2 += loss2.item()
                n_batches = batch_idx + 1
                pbar.set_postfix({
                    "loss": f"{epoch_loss / n_batches:.4f}",
                    "E|Y-Ŷ|": f"{epoch_loss1 / n_batches:.4f}",
                    "E|Ŷ-Ŷ'|": f"{epoch_loss2 / n_batches:.4f}",
                })

        # Final evaluation on original scale
        self.model.eval()
        x, y = self.unstandardize_data(y, x)
        self.tr_loss = self.eval_loss(x, y, loss_type="energy", verbose=True)

        if verbose:
            print(f"\nTraining loss (original scale):\n"
                  f"  energy_loss: {self.tr_loss[0]:.4f}\n"
                  f"  E|Y-Yhat|:  {self.tr_loss[1]:.4f}\n"
                  f"  E|Yhat-Yhat'|: {self.tr_loss[2]:.4f}")

    @torch.no_grad()
    def predict(self, x, target="mean", sample_size=100):
        """Point prediction.

        Args:
            x (torch.Tensor): predictors.
            target: "mean", "median", float (quantile), or list. Default "mean".
            sample_size (int): samples for Monte Carlo. Default 100.

        Returns:
            torch.Tensor or list.
        """
        self.eval_mode()
        x = vectorize(x).to(self.device)
        x = self.standardize_data(x)
        y_pred = self.model.predict(x, target, sample_size)
        if isinstance(y_pred, list):
            return [self.unstandardize_data(yp) for yp in y_pred]
        return self.unstandardize_data(y_pred)

    @torch.no_grad()
    def sample(self, x, sample_size=100, expand_dim=True):
        """Sample from the learned conditional distribution.

        Args:
            x (torch.Tensor): predictors.
            sample_size (int): samples per input. Default 100.
            expand_dim (bool): if True, return (n, d_y, S). Default True.

        Returns:
            torch.Tensor.
        """
        self.eval_mode()
        x = vectorize(x).to(self.device)
        x = self.standardize_data(x)
        y_samples = self.model.sample(x, sample_size, expand_dim=expand_dim)
        y_samples = self.unstandardize_data(y_samples, expand_dim=expand_dim)
        if sample_size == 1:
            y_samples = y_samples.squeeze(-1)
        return y_samples

    @torch.no_grad()
    def eval_loss(self, x, y, loss_type="l2", sample_size=None, beta=1, verbose=False):
        """Evaluate loss on data.

        Args:
            x (torch.Tensor): predictors.
            y (torch.Tensor): responses.
            loss_type (str): "l2", "l1", "energy", or "cor". Default "l2".
            sample_size (int): samples. Default None (auto).
            beta (float): energy loss power. Default 1.
            verbose (bool): return loss components. Default False.

        Returns:
            float or tuple.
        """
        if sample_size is None:
            sample_size = 2 if loss_type == "energy" else 100
        self.eval_mode()
        x = vectorize(x).to(self.device)
        y = vectorize(y).to(self.device)
        if loss_type == "l2":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = (y - y_pred).pow(2).mean()
        elif loss_type == "cor":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = cor(y, y_pred)
        elif loss_type == "l1":
            y_pred = self.predict(x, target=0.5, sample_size=sample_size)
            loss = (y - y_pred).abs().mean()
        else:
            assert loss_type == "energy"
            y_samples = self.sample(x, sample_size=sample_size, expand_dim=False)
            loss = energy_loss(y, y_samples, beta=beta, verbose=verbose)
        if not verbose:
            return loss.item()
        loss, loss1, loss2 = loss
        return loss.item(), loss1.item(), loss2.item()

    @torch.no_grad()
    def plot(self, x_te, y_te, x_tr=None, y_tr=None, x_idx=0, y_idx=0,
             target="mean", sample_size=100, save_dir=None,
             alpha=0.8, ymin=None, ymax=None):
        """Plot predictions vs true data.

        Args:
            x_te, y_te: test data.
            x_tr, y_tr: optional training data.
            x_idx, y_idx: which dimensions to plot.
            target: prediction target.
            sample_size: samples for prediction.
            save_dir: path to save figure.
            alpha: transparency for sampled points.
            ymin, ymax: y-axis limits.
        """
        if x_tr is not None and y_tr is not None:
            x_tr = vectorize(x_tr)
            y_tr = vectorize(y_tr)
            plt.scatter(x_tr[:, x_idx].cpu(), y_tr[:, y_idx].cpu(),
                        s=1, label="training data", color="silver")
            plt.scatter(x_te[:, x_idx].cpu(), y_te[:, y_idx].cpu(),
                        s=1, label="test data", color="gold")
            x = torch.cat((x_tr, x_te), dim=0)
            y = torch.cat((y_tr, y_te), dim=0)
        else:
            x_te = vectorize(x_te)
            y_te = vectorize(y_te)
            plt.scatter(x_te[:, x_idx].cpu(), y_te[:, y_idx].cpu(),
                        s=1, label="true data", color="silver")
            x = x_te
            y = y_te
        x = x.to(self.device)

        if target != "sample":
            y_pred = self.predict(x, target=target, sample_size=sample_size)
            plt.scatter(x[:, x_idx].cpu(), y_pred[:, y_idx].cpu(),
                        s=1, label="predictions", color="lightskyblue")
        else:
            y_sample = self.sample(x, sample_size=sample_size, expand_dim=False)
            x_rep = x.repeat(sample_size, 1)
            plt.scatter(x_rep[:, x_idx].cpu(), y_sample[:, y_idx].cpu(),
                        s=1, label="samples", color="lightskyblue", alpha=alpha)
        plt.legend(markerscale=2)
        plt.ylim(ymin, ymax)
        xlabel = r"$x$" if x.shape[1] == 1 else rf"$x_{{{x_idx}}}$"
        ylabel = r"$y$" if y.shape[1] == 1 else rf"$y_{{{y_idx}}}$"
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if save_dir is not None:
            make_folder(save_dir)
            plt.savefig(save_dir, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def save(self, path):
        """Save fitted model to disk.

        Args:
            path (str): file path (e.g., "model.pt").
        """
        state = {
            "model_state_dict": self.model.state_dict(),
            "hyperparams": {
                "in_dim": self.model.in_dim, "out_dim": self.model.out_dim,
                "num_layer": self.num_layer, "hidden_dim": self.hidden_dim,
                "noise_dim": self.noise_dim, "resblock": self.resblock,
                "add_bn": self.add_bn, "classification": self.classification,
                "out_act": self.out_act,
            },
            "standardization": {
                "x_mean": self.x_mean, "x_std": self.x_std,
                "y_mean": self.y_mean, "y_std": self.y_std,
            },
            "training_state": {
                "lr": self.lr, "beta": self.beta,
                "standardize": self.standardize,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
            },
            "tr_loss": self.tr_loss,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, device=None):
        """Load a fitted model from disk.

        Args:
            path (str): file path.
            device: target device. Default None (auto-detect).

        Returns:
            Engressor: fitted model ready for prediction.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        hp = state["hyperparams"]
        ts = state["training_state"]
        eng = cls(
            device=device, verbose=False,
            lr=ts["lr"], beta=ts["beta"],
            standardize=ts["standardize"],
            num_epochs=ts["num_epochs"],
            batch_size=ts["batch_size"],
            **hp)
        eng.model.load_state_dict(state["model_state_dict"])
        eng.model.to(eng.device)
        for k, v in state["standardization"].items():
            if v is not None:
                v = v.to(eng.device)
            setattr(eng, k, v)
        eng.tr_loss = state["tr_loss"]
        eng.model.eval()
        return eng
