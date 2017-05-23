function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%convert y(i) from 1-10 to [0 1 0 0 0 ..] 
y_new = zeros(m, num_labels);
mm = 1:num_labels;
for i = 1:m
   y_new(i, :) = mm==y(i);
endfor


na1 = [ones(m, 1) X];
z2 = na1 * Theta1';
a2 = sigmoid(z2);

na2 = [ones(size(a2,1), 1) a2];
h = sigmoid(na2 * Theta2');

for i=1:m
    for k=1:num_labels
        J += -y_new(i,k)*log(h(i,k)) - (1-y_new(i,k))*log(1-h(i,k));
    endfor
endfor

J = J/m;

Theta1_sq = Theta1 .^ 2;
Theta1_sq(:, 1) = 0;
Theta2_sq = Theta2 .^ 2;
Theta2_sq(:, 1) = 0;
J += lambda/(2*m) * (sum(sum(Theta1_sq, 2))+sum(sum(Theta2_sq, 2)));

%[dumb, p] = max(a3, [], 2);

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

delta3 = h - y_new;
v1 = delta3*Theta2;
v2 = [ones(m,1) sigmoidGradient(z2)];
delta2 = v1 .* v2;
delta2 = delta2(:, 2:end);

Theta1_grad = delta2'*na1/m;
Theta2_grad = delta3'*na2/m;

Theta1_grad += lambda/m * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad += lambda/m * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

grad = [Theta1_grad(:); Theta2_grad(:)];
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
