from pathlib import Path

import pytest

from rag_modules.data_preparation import DataPreparationModule


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


def load(root, **kwargs):
    module = DataPreparationModule(str(root), **kwargs)
    module.load_documents()
    module.chunk_documents()
    return module


def test_ids_stable_after_reload_and_unrelated_document_added(tmp_path):
    write(tmp_path, 'z.md', '# 手册\n说明\n## 操作\n启动服务')
    first = load(tmp_path)
    ids = [c.metadata['chunk_id'] for c in first.chunks]
    write(tmp_path, 'a.md', '# 无关文档\n另一篇')
    second = load(tmp_path)
    assert ids == [c.metadata['chunk_id'] for c in second.chunks if c.metadata['source'] == 'z.md']
    assert first.documents[0].metadata['document_id'] == second.documents[1].metadata['document_id']
    assert all(second.parent_child_map[c.metadata['chunk_id']] == c.metadata['document_id'] for c in second.chunks)


def test_content_change_changes_chunk_not_document_id(tmp_path):
    write(tmp_path, 'a.md', '# 标题\n旧正文')
    before = load(tmp_path)
    write(tmp_path, 'a.md', '# 标题\n新正文')
    after = load(tmp_path)
    assert before.documents[0].metadata['document_id'] == after.documents[0].metadata['document_id']
    assert before.chunks[0].metadata['chunk_id'] != after.chunks[0].metadata['chunk_id']


def test_skip_templates_empty_comments_and_non_markdown(tmp_path):
    for name, text in [('template/a.md', '模板'), ('templates/b.md', '模板'),
                       ('示例菜.md', '模板'), ('empty.md', ' \n'),
                       ('comment.md', '<!-- 说明 -->'), ('a.txt', '文本')]:
        write(tmp_path, name, text)
    write(tmp_path, 'valid.MD', '# 普通手册\n<!-- 注释 -->\n正文 ![图示](image.png)')
    module = load(tmp_path)
    assert len(module.documents) == 1
    chunk = module.chunks[0]
    assert chunk.metadata['title'] == '普通手册'
    assert chunk.metadata['section'] == '普通手册'
    assert '<!--' not in chunk.page_content and 'image.png' not in chunk.page_content
    assert '<!--' in module.documents[0].page_content  # raw source retained
    assert not {'dish_name', 'category', 'difficulty'} & chunk.metadata.keys()


def test_same_text_different_sources_have_distinct_ids(tmp_path):
    write(tmp_path, 'a.md', '相同文本')
    write(tmp_path, 'b.md', '相同文本')
    module = load(tmp_path)
    assert len({c.metadata['chunk_id'] for c in module.chunks}) == 2
    assert all(c.metadata['section'] for c in module.chunks)


def test_recipe_metadata_is_opt_in(tmp_path):
    write(tmp_path, 'vegetable_dish/a.md', '# 菜谱\n预估烹饪难度：★★')
    assert 'category' not in load(tmp_path).chunks[0].metadata
    metadata = load(tmp_path, recipe_metadata_enabled=True).chunks[0].metadata
    assert metadata['category'] == '素菜' and metadata['difficulty'] == '简单'


def test_empty_corpus_and_repeated_chunking(tmp_path):
    module = load(tmp_path)
    assert module.chunks == []
    write(tmp_path, 'a.md', '有内容')
    module.load_documents()
    module.chunk_documents()
    module.documents = []
    assert module.chunk_documents() == []
    assert module.parent_child_map == {}


def test_unreadable_document_stops_build(tmp_path):
    (tmp_path / 'invalid.md').write_bytes(b'\xff\xff')
    with pytest.raises(ValueError, match='停止建库'):
        load(tmp_path)
