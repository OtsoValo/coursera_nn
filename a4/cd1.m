function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

	visible_data = sample_bernoulli(visible_data);
	
	num_configs = size(visible_data, 2);

    hidden_prob1 = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    hidden1 = sample_bernoulli(hidden_prob1);

    visible_prob2 = hidden_state_to_visible_probabilities(rbm_w, hidden1);
    visible2 = sample_bernoulli(visible_prob2);

    hidden_prob2 = visible_state_to_hidden_probabilities(rbm_w, visible2);
    hidden2 = sample_bernoulli(hidden_prob2);

    ret = zeros(size(rbm_w));

    first = 1/num_configs * hidden1 * visible_data';

    second = 1/num_configs * hidden_prob2 * visible2';

    ret = first - second;

end
