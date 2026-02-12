async def agent_loop():
    agent_prompt = load_md("agent.md")
    skills = load_all_skills("skills/")
    
    while True:
        # 1. Gather context
        portfolio = await get_portfolio_state()
        market_data = await get_market_overview()
        signals = await run_research_pipeline()
        pending_commands = await check_telegram_commands()
        
        # 2. Build Claude message
        messages = [
            {"role": "user", "content": f"""
            Current portfolio: {json.dumps(portfolio)}
            Market overview: {json.dumps(market_data)}
            New signals: {json.dumps(signals)}
            User commands: {json.dumps(pending_commands)}
            
            Based on your agent.md directives and skills, 
            what actions should we take? Respond with structured JSON.
            """}
        ]
        
        # 3. Call Claude with tool use
        response = await claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",  # Sonnet for routine
            system=agent_prompt + "\n\n" + skills,
            messages=messages,
            tools=TOOL_DEFINITIONS,  # exchange, research, portfolio tools
            max_tokens=4096
        )
        
        # 4. Execute tool calls with safety checks
        for tool_call in response.tool_use_blocks:
            if tool_call.name == "execute_trade":
                if passes_risk_checks(tool_call.input):
                    if requires_approval(tool_call.input):
                        await request_telegram_approval(tool_call)
                    else:
                        await execute_safely(tool_call)
            # ... handle other tools
        
        # 5. Log everything
        await log_to_db(response, actions_taken)
        
        await asyncio.sleep(300)  # 5 min cycle