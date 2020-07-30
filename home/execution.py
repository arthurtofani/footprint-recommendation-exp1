def evaluation_instances():
  instances = []
  itr = 1
  term_field = 'track'
  for query_terms_ratio in [0.4, 0.2]:
    for document_field in ['session']:
      for memory_size in [1, 2, 3, 4]:
        instances.append([itr, term_field, document_field, query_terms_ratio, memory_size])
  return instances

