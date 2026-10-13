from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableLambda
from config import RAGConfig
from main import MarkdownRAGSystem
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.generation_integration import GenerationIntegrationModule, NO_EVIDENCE


def app_with_hit(tmp_path):
    (tmp_path / 'manual.md').write_text('# 通用指南\n## 前言\n' + '背景' * 2000
        + '\n## 恢复\n恢复命令为 restore --safe。', encoding='utf-8')
    data = DataPreparationModule(str(tmp_path))
    data.load_documents()
    chunks = data.chunk_documents()
    hit = next(c for c in chunks if 'restore --safe' in c.page_content)
    app = MarkdownRAGSystem(RAGConfig(data_path=str(tmp_path)))
    app.data_module = data
    app.index_module = Mock()
    app.index_module.vectorstore.semantic_similarities.return_value = [0.9]
    app.retrieval_module = Mock()
    app.retrieval_module.retrieve.return_value = [hit]
    prompts = []
    def llm(prompt):
        prompts.append(prompt.to_string())
        return '执行 restore --safe [S1]'
    app.generation_module = GenerationIntegrationModule(llm=RunnableLambda(llm))
    return app, hit, prompts


def test_main_passes_hit_not_long_parent_and_returns_sources(tmp_path):
    app, hit, prompts = app_with_hit(tmp_path)
    app.generation_module.query_rewrite = Mock(side_effect=AssertionError('must be disabled'))
    result = app.ask_question('如何恢复？')
    assert result.sources[0].chunk_id == hit.metadata['chunk_id']
    assert result.sources[0].content == hit.page_content
    assert 'restore --safe' in prompts[0]
    assert '背景' not in prompts[0]
    app.generation_module.query_rewrite.assert_not_called()


@pytest.mark.parametrize('rewrite', [RuntimeError('network unavailable'), '', '  ', None, 'recovery command'])
def test_optional_rewrite_fallback_and_original_question(tmp_path, rewrite):
    app, hit, prompts = app_with_hit(tmp_path)
    app.config.query_rewrite_enabled = True
    app.generation_module.query_rewrite = Mock()
    if isinstance(rewrite, Exception):
        app.generation_module.query_rewrite.side_effect = rewrite
    else:
        app.generation_module.query_rewrite.return_value = rewrite
    app.ask_question('如何恢复？')
    expected = rewrite if rewrite == 'recovery command' else '如何恢复？'
    app.retrieval_module.retrieve.assert_called_once_with(expected, mode='hybrid', final_k=5)
    assert '如何恢复？' in prompts[0]


def test_filters_only_explicit_and_no_recipe_routing(tmp_path):
    app, hit, prompts = app_with_hit(tmp_path)
    app.ask_question('简单的系统恢复方法')
    app.retrieval_module.retrieve.assert_called_once_with('简单的系统恢复方法', mode='hybrid', final_k=5)
    app.retrieval_module.retrieve.reset_mock()
    app.retrieval_module.retrieve.return_value = [hit]
    app.ask_question('恢复', filters={'source': 'manual.md'})
    app.retrieval_module.retrieve.assert_called_once_with('恢复', filters={'source': 'manual.md'}, mode='hybrid', final_k=5)
    assert not hasattr(app.generation_module, 'query_router')


def test_empty_results_normal_and_stream_do_not_call_llm(tmp_path):
    app, _, prompts = app_with_hit(tmp_path)
    app.retrieval_module.retrieve.return_value = []
    assert app.ask_question('不存在的内容').to_dict() == {'answer': NO_EVIDENCE, 'sources': []}
    events = list(app.ask_question('不存在的内容', stream=True))
    assert events[0].sources == [] and events[1].text == NO_EVIDENCE
    assert prompts == []


def test_relevance_gate_rejects_low_similarity_without_calling_llm(tmp_path):
    app, _, prompts = app_with_hit(tmp_path)
    app.index_module.vectorstore.semantic_similarities.return_value = [0.30]

    result = app.ask_question('完全不相关的问题')
    assert result.to_dict() == {'answer': NO_EVIDENCE, 'sources': []}
    events = list(app.ask_question('完全不相关的问题', stream=True))
    assert events[0].sources == [] and events[1].text == NO_EVIDENCE
    assert prompts == []


def test_relevance_gate_can_be_disabled(tmp_path):
    app, _, prompts = app_with_hit(tmp_path)
    app.config.retrieval_relevance_threshold = None
    app.index_module.vectorstore.semantic_similarities.side_effect = AssertionError('gate disabled')

    result = app.ask_question('如何恢复？')
    assert result.sources and prompts


def test_empty_question_rejected(tmp_path):
    app, _, _ = app_with_hit(tmp_path)
    with pytest.raises(ValueError, match='不能为空'):
        app.ask_question('  ')
    app.retrieval_module.retrieve.assert_not_called()


def test_llm_setup_uses_openai_compatible_chat_completions(monkeypatch):
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv('LLM_API_KEY', 'test-key')
    monkeypatch.setenv('LLM_BASE_URL', 'https://example.invalid/v1')
    monkeypatch.setenv('LLM_MODEL', 'test-model')
    module = GenerationIntegrationModule()
    module.setup_llm()

    assert isinstance(module.llm, ChatOpenAI)
    assert module.llm.model_name == 'test-model'
    assert module.llm.openai_api_base == 'https://example.invalid/v1'
    assert module.llm.use_responses_api is False
