
create table ai.vectorizer
( id int not null primary key generated by default as identity
, source_schema name not null
, source_table name not null
, source_pk jsonb not null
, target_schema name not null
, target_table name not null
, view_schema name not null
, view_name name not null
, trigger_name name not null
, queue_schema name
, queue_table name
, config jsonb not null
, unique (target_schema, target_table)
);
perform pg_catalog.pg_extension_config_dump('ai.vectorizer'::pg_catalog.regclass, '');
perform pg_catalog.pg_extension_config_dump('ai.vectorizer_id_seq'::pg_catalog.regclass, '');

-- TODO: add table for items that failed to be embedded
