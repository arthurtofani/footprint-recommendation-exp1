import footprint.clients as db

def connect_to_elasticsearch(p, db_index_name, clear=True):
  cli = db.elasticsearch.Connection(host='elasticsearch', port=9200)
  if clear:
    cli.clear_index(db_index_name)
    cli.setup_index(db_index_name, initial_settings())
  p.set_connection(cli)



def initial_settings():
  return {
    "settings" : {
      "analysis" : {
        "analyzer" : {
          "tokens_by_spaces": {
            "tokenizer": "divide_tokens_by_spaces"
          }
        },
        "tokenizer": {
          "divide_tokens_by_spaces": {
            "type": "simple_pattern_split",
            "pattern": " "
          }
        }
      }
    }
  }
