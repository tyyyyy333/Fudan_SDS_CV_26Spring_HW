from hw2_cv.utils import log_info, prepare_run, save_json


def run_training(config):
    from ultralytics import YOLO

    config, output_dir, _ = prepare_run(config)
    train_cfg = config["train"]
    project = train_cfg.get("project", str(output_dir / "train"))
    name = train_cfg.get("name", config["profile"])

    log_info(
        f"[task2-train] model={train_cfg['model']} | epochs={train_cfg['epochs']} | "
        f"imgsz={train_cfg['imgsz']} | batch={train_cfg['batch']} | profile={config['profile']}"
    )

    model = YOLO(train_cfg["model"])
    results = model.train(
        data=train_cfg["data"],
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        device=train_cfg.get("device", 0),
        workers=train_cfg.get("workers", 4),
        project=project,
        name=name,
        seed=config.get("seed", 42),
        exist_ok=True,
        patience=train_cfg.get("patience", 50),
        cache=train_cfg.get("cache", False),
        pretrained=train_cfg.get("pretrained", True),
        **train_cfg.get("overrides", {}),
    )
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    best_path = getattr(trainer, "best", None)
    summary = {
        "output_dir": str(output_dir),
        "results": str(results) if results is not None else None,
        "save_dir": str(save_dir) if save_dir is not None else None,
        "best_path": str(best_path) if best_path is not None else None,
        "profile": config["profile"],
        "model": train_cfg["model"],
    }
    save_json(summary, output_dir / "summary.json")
    log_info(
        f"[task2-train] done | save_dir={summary['save_dir']} | best={summary['best_path']}"
    )
    return summary


def run_yolo_training(config):
    return run_training(config)
