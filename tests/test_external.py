from hw3cv.external import EXTERNAL_REPOSITORIES, bootstrap_commands


def test_external_repository_urls_include_required_frameworks():
    assert EXTERNAL_REPOSITORIES["2d-gaussian-splatting"].url == "https://github.com/hbb1/2d-gaussian-splatting.git"
    assert EXTERNAL_REPOSITORIES["threestudio"].url == "https://github.com/threestudio-project/threestudio.git"
    assert EXTERNAL_REPOSITORIES["Magic123"].url == "https://github.com/guochengqian/Magic123.git"
    assert EXTERNAL_REPOSITORIES["lerobot"].url == "https://github.com/huggingface/lerobot.git"
    assert EXTERNAL_REPOSITORIES["calvin"].url == "https://github.com/mees/calvin.git"
    assert EXTERNAL_REPOSITORIES["tiny-cuda-nn"].url == "https://github.com/NVlabs/tiny-cuda-nn.git"


def test_bootstrap_commands_clone_only_when_missing(tmp_path):
    commands = bootstrap_commands(tmp_path)

    assert commands[0] == ["mkdir", "-p", str(tmp_path)]
    assert [
        "git",
        "clone",
        "https://github.com/hbb1/2d-gaussian-splatting.git",
        str(tmp_path / "2d-gaussian-splatting"),
    ] in commands

    (tmp_path / "Magic123").mkdir()
    commands = bootstrap_commands(tmp_path)

    assert not any(command[-1] == str(tmp_path / "Magic123") for command in commands if command[:2] == ["git", "clone"])
