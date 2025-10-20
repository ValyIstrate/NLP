def find_context_for_question(question, data):
    match = data[data['question'].str.lower().str.strip() == question.lower().strip()]

    if not match.empty:
        return match.iloc[0]['text']
    else:
        from difflib import get_close_matches
        questions = data['question'].dropna().tolist()
        closest = get_close_matches(question, questions, n=1, cutoff=0.7)
        if closest:
            print(f"No exact match found. Using closest match: '{closest[0]}'")
            context = data.loc[data['question'] == closest[0], 'text'].iloc[0]
            return context
        else:
            print("No similar question found in dataset.")
            return None