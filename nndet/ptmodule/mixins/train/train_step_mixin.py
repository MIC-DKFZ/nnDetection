import torch


class TrainMixin:
    def training_step(self, batch, batch_idx):
        """
        Computes a single training step
        See :class:`DETRBase` for more information
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)

        targets = {key: item for key, item in batch.items() if "target_" in key}
        if "target_seg" in targets:
            # [optional] add semantic segmentation to targets if available
            # Remove channel dimension of semantic segmentation
            targets["target_seg"] = targets["target_seg"][:, 0]
        if "target_binary_masks" in targets:
            # [optional] add binary masks to targets if available
            targets["target_binary_masks"] = targets["target_binary_masks"]

        losses = self.model.train_step(
            images=batch["data"],
            targets=targets,
            batch_num=batch_idx,
        )
        weight_dict = self.model.head.weight_dict
        norm_dec_layers = self.model.decoder_layers if self.model.head.aux_loss else 1

        loss = 0
        # if the weight dict has standard format, use the same weights for all auxiliary losses
        if len(weight_dict) == 3:
            weight_sum = sum(weight_dict.values())
            for key, value in losses.items():
                # split for auxiliary losses
                weight_key = "_".join(key.split("_")[:2])
                if weight_key in weight_dict:
                    loss += (
                        value * weight_dict[weight_key] / (norm_dec_layers * weight_sum)
                    )
        # Else the weight dict will contain all necessary keys for loss computation
        else:
            weight_sum = 0
            for key, loss_weight in weight_dict.items():
                # if the key has more than one _ it is an auxiliary or de-noising loss, and we slowly reduce the weight
                # of these after 10 epochs
                if key not in losses:
                    continue
                if (
                    self.model.decrease_aux_loss
                    and len(key.split("_")) > 2
                    and self.current_epoch + 1 > 10
                ):
                    loss += loss_weight / ((self.current_epoch + 1) / 10) * losses[key]
                    weight_sum += loss_weight / ((self.current_epoch + 1) / 10)

                else:
                    loss += loss_weight * losses[key]
                    weight_sum += loss_weight
            loss /= weight_sum
        return {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}

    def validation_step(self, batch, batch_idx):
        """
        Computes a single validation step (same as train step but with
        additional prediction processing)
        See ::class::`BaseRetinaNet` for more information
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)

            targets = {key: item for key, item in batch.items() if "target_" in key}
            if "target_seg" in targets:
                # [optional] add semantic segmentation to targets if available
                # Remove channel dimension of semantic segmentation
                targets["target_seg"] = targets["target_seg"][:, 0]
            if "target_binary_masks" in targets:
                # [optional] add bianry masks to targets if available
                targets["target_binary_masks"] = targets["target_binary_masks"]

            losses, predictions = self.model.validation_step(
                images=batch["data"],
                targets=targets,
                batch_num=batch_idx,
            )
            weight_dict = self.model.head.weight_dict
            num_dec_layers = self.model.decoder_layers

            loss = 0
            for key, value in losses.items():
                # split for auxiliary losses
                weight_key = "_".join(key.split("_")[:2])
                if weight_key in weight_dict:
                    loss += value * weight_dict[weight_key]
            weight_sum = sum(weight_dict.values())
            loss /= num_dec_layers * weight_sum

        # self.log_dict(losses, prog_bar=True)

        super().evaluation_step(predictions=predictions, targets=targets)

        return {
            "loss": loss.detach().item(),
            **{key: l.detach().item() for key, l in losses.items()},
        }
