import pytest

@pytest.mark.asyncio
async def test_passive_user_feedback_survives_resume_without_starting_turn(context):
    await context.add_message({'role':'user','content':'A task'})
    await context.add_message({'role':'assistant','content':'A result'})
    await context.add_message({'role':'user','content':'User reacted ❤️ to the result','metadata':{'passive':True,'ephemeral':True,'persisted':True}})
    assert context._current_turn==1
    request=await context.get_messages_for_request()
    assert any('reacted ❤️' in str(row.get('content')) for row in request)
    rows=await context.get_messages()
    from amplifier_module_context_managed import ManagedContextManager
    resumed=ManagedContextManager(session_dir=context._session_dir)
    await resumed._load_from_transcript()
    assert resumed._current_turn==1
    await resumed.add_message({'role':'user','content':'Continue'})
    assert resumed._current_turn==2
