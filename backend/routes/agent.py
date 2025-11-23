@router.post("/agent/search_code")
def agent_search_code(query: str = Body(...)):
    results = ripgrep_search(root=REPO_ROOT, query=query)
    return {"matches": results}

@router.post("/agent/read_file")
def agent_read_file(payload=Body(...)):
    path = sanitize_path(payload["path"])
    with open(path, "r", encoding="utf8") as f:
        return {"content": f.read()}

@router.post("/agent/apply_patch")
def agent_apply_patch(payload=Body(...)):
    patch = payload["diff"]
    result = apply_unified_diff(patch)
    return {"result": result}
