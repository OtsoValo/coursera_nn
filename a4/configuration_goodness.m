function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
	num_configs = size(visible_state, 2);
	num_hidden = size(hidden_state, 1);
	num_visible = size(visible_state, 1);

    total = 0;
    for x = 1:num_configs
    	for y = 1:num_visible
    		for z = 1:num_hidden
    			total += visible_state(y, x) * hidden_state(z, x) * rbm_w(z, y);
    		end
    	end
    end

    G = total/num_configs;
end
