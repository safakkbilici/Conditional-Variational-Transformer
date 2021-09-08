def is_cheating(generated_sentence, real_sentence, gram = 1, check_levenshtein = False, print_it = False):
    generated_sentence = generated_sentence.lower()
    real_sentence = real_sentence.lower()
    
    if generated_sentence == real_sentence:
        return "Same sentence!"
    if check_levenshtein:
        levenshtein = Levenshtein()
        distance = levenshtein.distance(generated_sentence, real_sentence)
    
    count = 0
    grams = []
    generated_sentence_splitted = generated_sentence.split()
    for i in range(len(generated_sentence_splitted) - gram + 1):
        grams.append(generated_sentence_splitted[i:i+gram])
        
    for ngram in grams:
        gram_sentence = " ".join(ngram)
        if gram_sentence in real_sentence:
            if print_it:
                print("Fount the gram: {}".format(gram_sentence))
            count += 1
    if check_levenshtein:
        return count, distance
    return count

def cheating(df_generated, df_real, feature_name, gram, check_levenshtein = False, print_it = False):
    generated_sentences = getattr(df_generated, feature_name)
    real_sentences = getattr(df_real, feature_name)
    
    general_count = 0
    with tqdm(total = len(df_generated) * len(df_real)) as tt:
        for idx, generated_sentence in enumerate(generated_sentences):
            local_count = 0
            for real_sentece in df_imdb.sentence:
                a = is_cheating(generated_sentence, real_sentece, gram, print_it)
                general_count += a
                local_count += a
                tt.update()

            report = f"Generated Sentence ID {idx}: "
            report += f"found {local_count} matching for n-gram: {gram} "
            report += f"-> len: {len(generated_sentence.split())}"
            print(report)
