def get_span(filename):
    file = open(filename, "r")

    count = 0
    prev_span = None
    categories = []
    for line in file:
        tabs = line.split("\t")
        span = tabs[0]
        if span != prev_span:
            count += 1
            if prev_span is not None:
                yield (prev_span, categories) # send span and categories
            prev_span = span
            categories = []
        categories.append(tabs[1][:-1]) # get rid of \n
    
    if prev_span is not None:
        yield (prev_span, categories) # get last span
    file.close()
    return None # No more iterations

def translate_categories(translator_filename):
    file = open(translator_filename, "r")
    _dict = {}
    for line in file:
        tabs = line.split("\t")
        _dict[tabs[0]] = tabs[1]
    file.close()
    return _dict


def main(filename, translator_filename, output_filename):
    output_file = open(output_filename, "w")

    subcats_to_cats = translate_categories(translator_filename)

    for _tuple in get_span(filename):
        span = _tuple[0]
        subs = _tuple[1]

        sub_data = {}
        amb_data = {}
        cat_data = {}
        amb_count = 0
        cat_count = 0
        sub_count = len(subs)

        for sub in subs:
            if sub not in subcats_to_cats:
                raise Exception(sub + " missing from dict")
            cat = subcats_to_cats[sub]

            #check which dictionary cat belongs to
            if cat in AMBIGUOUS_CATS:
                _dict = amb_data
                amb_count += 1
            else:
                _dict = cat_data
                cat_count += 1
            
            # add cat to dictionary
            if cat not in _dict:
                _dict[cat] = 1
            else:
                _dict[cat] += 1

            if sub not in sub_data:
                sub_data[sub] = 1
            else:
                sub_data[sub] += 1

        # percentages
        for i in sub_data:
            sub_data[i] /= sub_count
        
        for i in amb_data:
            amb_data[i] /= amb_count

        for i in cat_data:
            cat_data[i] /= cat_count

        sub_data = sorted(sub_data.items(), key=lambda x: x[1], reverse=True)
        amb_data = sorted(amb_data.items(), key=lambda x: x[1], reverse=True)
        cat_data = sorted(cat_data.items(), key=lambda x: x[1], reverse=True)

        output_file.write(span + "\t" + str(cat_data) + "\t" + str(amb_data) + "\t" + str(sub_data) + "\n")



AMBIGUOUS_CATS = ['Activity', 'Product', 'Procedure', 'Finding', 'Concept']

if __name__ == "__main__":
    main("/home/edan/for_edan/cofie_log.tsv",
         "/home/edan/projects/DyGIE-MECHANIC/spacy_cats/categories_final.tsv",
         output_filename="dataset.tsv")