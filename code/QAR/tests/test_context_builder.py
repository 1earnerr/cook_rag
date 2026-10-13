import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from rag_modules.generation_integration import GenerationIntegrationModule, NO_EVIDENCE


def doc(content, number=1, title='标题', section='正文'):
    return Document(page_content=content, metadata=dict(chunk_id=f'c{number}',
        document_id=f'd{number}', title=title, section=section, source=f'{number}.md'))


def test_long_first_hit_preserves_later_short_hit():
    module = GenerationIntegrationModule(context_max_chars=120)
    context, sources = module._build_context([doc('长' * 1000), doc('后续证据', 2)])
    assert len(context) <= 120
    assert [s.chunk_id for s in sources] == ['c1', 'c2']
    assert sources[1].content == '后续证据'
    assert context == '\n\n'.join(s.render() for s in sources)
    assert sources[0].content == '长' * len(sources[0].content)


@pytest.mark.parametrize('budget', [0, 1, 15, 30, 100, 2000])
def test_exact_budget_and_sources(budget):
    module = GenerationIntegrationModule()
    context, sources = module._build_context([doc('文字' * 1500), doc('另一证据', 2)], budget)
    assert len(context) <= budget
    assert context == '\n\n'.join(s.render() for s in sources)
    assert [s.evidence_id for s in sources] == [f'S{i+1}' for i in range(len(sources))]


def test_unfit_header_skipped_and_ids_contiguous():
    module = GenerationIntegrationModule(context_max_chars=100)
    context, sources = module._build_context([doc('正文', title='标题' * 200), doc('有效证据', 2)])
    assert len(sources) == 1 and sources[0].evidence_id == 'S1'
    assert '有效证据' in context


def test_dedupe_only_same_chunk_identity():
    module = GenerationIntegrationModule()
    context, sources = module._build_context([doc('相同正文'), doc('相同正文'), doc('相同正文', 2)])
    assert len(sources) == 2


def test_reject_untraceable_parent_document():
    with pytest.raises(ValueError, match='检索子块'):
        GenerationIntegrationModule()._build_context([Document(page_content='没有 ID')])


def test_empty_evidence_skips_llm_for_both_output_modes():
    def fail(_):
        pytest.fail('empty evidence must not invoke LLM')
    module = GenerationIntegrationModule(llm=RunnableLambda(fail))
    assert module.generate_basic_answer('问题', [doc('  ')]).to_dict() == {'answer': NO_EVIDENCE, 'sources': []}
    events = list(module.generate_basic_answer_stream('问题', []))
    assert [e.event for e in events] == ['sources', 'token', 'done']
    assert events[1].text == NO_EVIDENCE


def test_actual_prompt_and_returned_sources_are_identical():
    prompts = []
    def fake_llm(prompt):
        prompts.append(prompt.to_string())
        return '答案 [S1]'
    module = GenerationIntegrationModule(context_max_chars=100, llm=RunnableLambda(fake_llm))
    result = module.generate_basic_answer('如何启动?', [doc('启动命令 ' * 100)])
    assert result.answer == '答案 [S1]'
    assert result.sources[0].render() in prompts[0]
    assert '烹饪' not in prompts[0]
    assert set(result.sources[0].to_dict()) == {'evidence_id', 'chunk_id', 'document_id', 'title', 'section', 'source', 'content'}
    assert len(result.sources[0].content) < len('启动命令 ' * 100)
    events = list(module.generate_basic_answer_stream('如何启动?', [doc('启动命令')]))
    assert events[0].event == 'sources' and events[0].sources
    assert events[-1].event == 'done'
