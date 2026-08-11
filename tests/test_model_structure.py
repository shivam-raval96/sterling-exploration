from exploration.print_model_structure import human_size, shape_text


def test_human_size() -> None:
    assert human_size(1024) == "1.00 KiB"
    assert human_size(8_803_110_912 * 2).endswith("GiB")


def test_shape_text() -> None:
    assert shape_text((32, 4096)) == "32×4096"
    assert shape_text(()) == "scalar"
