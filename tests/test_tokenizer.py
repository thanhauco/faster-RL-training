from kvstreams.chat import ChatTemplate


def test_roundtrip(tok):
    text = "assign t3 to bob"
    assert tok.decode(tok.encode(text)) == text


def test_unknown_char_maps_to_unk(tok):
    assert tok.encode("é") == [tok.unk_id]


def test_chat_template_boundaries(tok):
    tpl = ChatTemplate(tok)
    sys = tpl.system("hi")
    assert sys[0] == tok.bos_id and sys[-1] == tok.end_id
    user = tpl.user_turn("q")
    assert user[0] == tok.role_id("user") and user[-1] == tok.role_id("assistant")
    assert all(tok.is_boundary(t) for t in (user[0], user[-2], user[-1]))


def test_padding_reaches_block_boundary(tok):
    tpl = ChatTemplate(tok)
    for n in range(40):
        assert (n + len(tpl.padding(n, 16))) % 16 == 0
    assert tpl.padding(32, 16) == []
