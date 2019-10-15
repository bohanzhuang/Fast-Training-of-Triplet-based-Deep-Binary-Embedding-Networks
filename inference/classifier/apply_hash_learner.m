

function bi_code=apply_hash_learner(feat_data, hash_learners, hash_learners_model)

assert(~isempty(hash_learners));

apply_hash_learner_fn=hash_learners_model.apply_hash_learner_fn;
bi_code=apply_hash_learner_fn(feat_data, hash_learners, hash_learners_model);

assert(isa(bi_code, 'int8'));


end



