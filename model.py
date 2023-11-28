import utils
import torch
import torch.nn as nn

from lightning import LightningModule

class ViT(nn.Module):
    def __init__(
        self,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        blocks: int = 4,
        mlp_head_units: list = [1024, 512],
        n_classes: int = 1,
        img_size: tuple = (224, 224),
        patch_size: tuple = (16, 16),
        n_channels: int = 3,
        d_model: int = 512,
    ):
        super().__init__()
        """
        Args:
            img_size: Size of the image
            patch_size: Size of the patch
            n_channels: Number of image channels
            d_model: The number of features in the transformer encoder
            nhead: The number of heads in the multiheadattention models
            dim_feedforward: The dimension of the feedforward network model in the encoder
            blocks: The number of sub-encoder-layers in the encoder
            mlp_head_units: The hidden units of mlp_head
            n_classes: The number of output classes
        """
        # self.img2seq = Img2Seq(img_size, patch_size, n_channels, d_model)
        self.patch_size = patch_size # (16, 16)
        nh, nw = img_size[0] // patch_size[0], img_size[1] // patch_size[1] # (14, 14)
        n_tokens = nh * nw # 196
        token_dim = patch_size[0] * patch_size[1] * n_channels # 768
        self.first_linear = nn.Linear(token_dim, d_model) # (768, 512)
        self.cls_token = nn.Parameter(torch.randn(1, d_model)) # (1, 512)
        self.pos_emb = nn.Parameter(utils.get_positional_embeddings(n_tokens, d_model)) # (196, 512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, blocks
        )
        self.mlp = utils.get_mlp(d_model, mlp_head_units, n_classes) # (512, [1024, 512], n_classes)

        self.classifer = nn.Sigmoid() if n_classes == 1 else nn.Softmax()

    def __call__(self, batch):
        """
        Shape:
            input: (b, c, h, w)
            output: (b, n_classes)
        """
        # batch = self.img2seq(batch) # (b, s, d)
        batch = torch.permute(batch, (0, 2, 3, 1)) # (b, h, w, c) = (b, 224, 224, 3)
        batch = utils.patchify(batch, self.patch_size) # (b, nh*nw, ph*pw*c) = (b, 196, 768)
        b = batch.shape[0]
        batch = self.first_linear(batch) # (b, nh*nw, d_model) = (b, 196, 512)
        cls = self.cls_token.expand([b, -1, -1]) # (b, 1, d_model) = (b, 1, 512)
        emb = batch + self.pos_emb # (b, nh*nw, d_model) = (b, 196, 512)
        batch = torch.cat([cls, emb], axis=1) # (b, nh*nw+1, d_model) = (b, 197, 512)

        batch = self.transformer_encoder(batch) # (b, s, d)
        batch = batch[:, 0, :] # (b, d)
        batch = self.mlp(batch) # (b, n_classes)
        output = self.classifer(batch) # (b, n_classes)
        return output

class ModelModule(LightningModule):
    def __init__(self, learning_rate: float = 1e-4, *args, model, **kwargs) -> None:
        '''
        Args:
            img_size: Size of the image

        Shape:
            input: (b, c, h, w)
            output: (b, n_classes)
        '''
        super().__init__()
        self.learing_rate = learning_rate
        self.model = model
        self.criteria = nn.CrossEntropyLoss()

        self.train_accuracy = []
        self.val_accuracy = []
        self.test_accuracy = []
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criteria(logits, y.long())
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        self.train_accuracy.append(accuracy)
        self.train_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criteria(logits, y.long())
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        self.val_accuracy.append(accuracy)
        self.val_loss.append(loss)
        return loss

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criteria(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log('test_accuracy', accuracy, prog_bar=True)
        self.log('test_loss', loss, prog_bar=True)
        self.test_accuracy.append(accuracy)
        self.test_loss.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        train_accuracy = torch.stack(self.train_accuracy).mean()
        train_loss = torch.stack(self.train_loss).mean()
        self.log('train_accuracy_epoch', train_accuracy)
        self.log('train_loss_epoch', train_loss)
        self.train_accuracy = []
        self.train_loss = []
        # print(f'Training Accuracy: {train_accuracy:.4f} Training Loss: {train_loss:.4f}')

    def on_validation_epoch_end(self) -> None:
        val_accuracy = torch.stack(self.val_accuracy).mean()
        val_loss = torch.stack(self.val_loss).mean()
        self.log('val_accuracy_epoch', val_accuracy)
        self.log('val_loss_epoch', val_loss)
        self.val_accuracy = []
        self.val_loss = []
        # epoch = self.current_epoch + 1
        # print(f'Epoch: {epoch} Validation Accuracy: {val_accuracy:.4f} Validation Loss: {val_loss:.4f}')

    def on_test_epoch_end(self) -> None:
        self.log('test_accuracy_epoch', torch.stack(self.test_accuracy).mean())
        self.log('test_loss_epoch', torch.stack(self.test_loss).mean())
        self.test_accuracy = []
        self.test_loss = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learing_rate)
