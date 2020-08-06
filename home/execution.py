def evaluation_instances():
  instances = []
  itr = 1
  for term_field in ['track']:
    for query_terms_ratio in [0.8, 0.5, 0.2, 0.6, 0.4]:
      for document_field in ['session', 'user']:
        for memory_size in [9, 10, 11, 12]:
          instances.append([itr, term_field, document_field, query_terms_ratio, memory_size])
  return instances

