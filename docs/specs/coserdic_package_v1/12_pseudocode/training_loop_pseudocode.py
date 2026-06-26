"""Pseudo-code for stage-wise CoSER-DiC training."""


def train_one_epoch(model, dataloader, optimizer, stage, loss_builder, logger):
    model.train()
    for batch in dataloader:
        x = batch["image"]
        out = model.forward_train(x, stage=stage)
        losses = loss_builder(stage=stage, x=x, out=out)
        total_loss = losses["total"]

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        logger.log({
            "loss_total": float(total_loss),
            "rate_s": float(out.rate_s),
            "rate_d": float(out.rate_d),
            "rate_total": float(out.rate_total),
            **{k: float(v) for k, v in losses.items() if k != "total"},
        })


def validate_with_actual_bpp(model, eval_images, metrics, bitstream_dir):
    results = []
    model.eval()
    for item in eval_images:
        x = item["image"]
        H, W = item["height"], item["width"]
        bitstream, metadata = model.compress(x)
        x_hat = model.decompress(bitstream)
        actual_bpp = 8 * len(bitstream) / (H * W)

        metric_values = metrics.compute(x, x_hat)
        results.append({
            "image_id": item["id"],
            "actual_bpp": actual_bpp,
            "bitstream_bytes": len(bitstream),
            **metric_values,
            **metadata.get("bpp_decomposition", {}),
        })

        save_bitstream(bitstream_dir, item["id"], bitstream)
    return results


def save_bitstream(bitstream_dir, image_id, bitstream):
    # placeholder
    pass
