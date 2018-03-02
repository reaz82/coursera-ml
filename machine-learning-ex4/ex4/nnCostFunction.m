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
J_i = 0;            % temporary variable to hold cost
X = [ones(m, 1) X]; % need to add column vector of 1s for bias 
  
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

  % z_2 is a 5000x25 matrix
  % a_2 is a 5000x26 matrix (includes bias)
  % a_3 is a 5000x10 matrix
  % X : m x [n+1] (5000 x 401)
  % Theta1 : h x [n+1] (25 x 401)
  z_2 = sigmoid(X * transpose(Theta1));
  a_2 = [ones(m, 1) z_2];
  a_3 = sigmoid(a_2 * transpose(Theta2));

  h_theta = a_3; % 5000x10 matrix holding class
  % develop a matrix y 
  y_k = eye(num_labels)(y,:);
  y = y_k;
for i = 1:m
  for j = 1:num_labels
    J = J + (-y(i,j).*log(h_theta(i,j)) - (1-y(i,j)).*log(1-h_theta(i,j)));
  end 
end 

J = J / m; 


% regularization code
% Theta1 and Theta2 can be of any size
% 3 layers but each layer can have any number of units
Theta1_sum = 0;
for i = 1:size(Theta1, 1)
  for j = 2:size(Theta1, 2)
    Theta1_sum = Theta1_sum + (Theta1(i,j) ^ 2);
  end
end

Theta2_sum = 0;
for i = 1:size(Theta2, 1)
  for j = 2:size(Theta2, 2)
    Theta2_sum = Theta2_sum + (Theta2(i,j) ^ 2);
  end
end

regularized_sum = (lambda/(2*m)) * (Theta1_sum + Theta2_sum);

J = J + regularized_sum;

% backpropagation 

d_2 = zeros(m,size(Theta1,1));
d_3 = zeros(m,num_labels);
Delta_1 = zeros(size(Theta1)); 
Delta_2 = zeros(size(Theta2));

% h_theta : m x k
% y_k : m x k
% d_3 : m x k

d_3 = h_theta - y;

% d_3 : m x k (5000 x 10)
% d_2 : m x h (5000 x 25)
% Theta2 : k x [h+1] (10 x 26)
% z_2 : m x h (5000 x 25)
d_2 = d_3*(Theta2(:,2:end)) .* sigmoidGradient(X * transpose(Theta1));

% Delta_1 is the product of d2 and a1
% Delta_1 : h x [n+1] (25 x 401) 
% d_2 : m x h (5000 x 25)
% a_1 = m x [n+1] (5000 x 401)
a_1 = X;

Delta_1 = transpose(d_2) * a_1;

% Delta_2 is the product of d3 and a2
% Delta_2 : k x [h+1] ( 10 x 26 )
% d_3 :  m x k (5000 x 10) 
% a_2 :  m x [h+1] (5000 x 26)

Delta_2 =  transpose(d_3) * a_2;

% first scale to get gradients
Theta1_grad = (m^-1) .* Delta_1;
Theta2_grad = (m^-1) .* Delta_2;

% apply regularization specifically
Theta1_grad(:,2:end) += (m^-1) .* (lambda .* Theta1(:,2:end));
Theta2_grad(:,2:end) += (m^-1) .* (lambda .* Theta2(:,2:end));




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
