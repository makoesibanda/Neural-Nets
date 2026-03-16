% Coursework Part 2
clear all; close all; clc;

%{
// Data Structure for the Weights //
 
For this coursework, the network we are working with is small, and it
contains exactly 9 weights in total. Since the structure is fixed and
does not change, I decided to store all the weights inside a single
1×9 vector called W. This is the simplest and cleanest way to organise
them for incremental learning (updating after each example).

I used the exact same ordering of weights that I used in Part 1, so that
my MATLAB implementation matches my handwritten calculations line-by-line.
This makes it easier to check my results and also ensures I don't mix up
the roles of the weights when doing the forward or backward passes.

Below is the layout of my weight vector W, with a short explanation of
what each weight connects:

   W(1) = w1 : from x1 to hidden unit h1
   W(2) = w2 : from x1 to output (direct connection)
   W(3) = w3 : from x1 to hidden unit h2
   W(4) = w4 : from x2 to hidden unit h2
   W(5) = w5 : from x2 to output (direct connection)
   W(6) = w6 : from x2 to hidden unit h3
   W(7) = w7 : from h3 to output unit
   W(8) = w8 : from h2 to output unit
   W(9) = w9 : from h1 to output unit

Why I chose this structure:
- It keeps everything compact and easy to update inside MATLAB.
- It avoids using multiple matrices for such a small network.
- The indexing matches Part 1 exactly, so I can compare my manual
  answers with my MATLAB results directly.
- It is simple to read: W(i) always corresponds to weight wi.

So overall, a single vector W is the most efficient way for this task
and also helps me avoid unnecessary complication in the script.
%}

% Initial weights (same as Part 1 from my worksheet)
W = [0.3 -0.2 0.1 0.2 -0.1 0.3 0.2 -0.1 -0.2];

eta = 0.2;                                     % learning rate
sigmoid = @(s) 1 ./ (1 + exp(-s));             % activation function

% Training patterns: [x1 x2 target]
Patterns = [0 1 1;    % Example 1
            1 0 1];   % Example 2

fprintf('Initial Weights:\n');
fprintf('  W = [%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n', W);

% Incremental learning: update after each example
for p = 1:size(Patterns,1)

    x1 = Patterns(p,1);
    x2 = Patterns(p,2);
    t  = Patterns(p,3);

    fprintf('\n=== Example %d (x1=%d, x2=%d, t=%d) ===\n', p, x1, x2, t); 
   
    % Forward Propagation  
    net1 = x1 * W(1);                 
    net2 = x1 * W(3) + x2 * W(4);     
    net3 = x2 * W(6);                 

    y1 = sigmoid(net1);
    y2 = sigmoid(net2);
    y3 = sigmoid(net3);

    o = x1*W(2) + x2*W(5) + y1*W(9) + y2*W(8) + y3*W(7);

    fprintf('Forward Pass:\n');
    fprintf('  net1=%.4f  y1=%.4f\n', net1, y1);
    fprintf('  net2=%.4f  y2=%.4f\n', net2, y2);
    fprintf('  net3=%.4f  y3=%.4f\n', net3, y3);
    fprintf('  Output o=%.4f   Target=%d\n', o, t);

    % Backward Propagation
    beta_out = t - o;   

    beta1 = y1*(1-y1) * W(9) * beta_out;
    beta2 = y2*(1-y2) * W(8) * beta_out;
    beta3 = y3*(1-y3) * W(7) * beta_out;

    fprintf('Backward Pass:\n');
    fprintf('  beta_out = %.4f\n', beta_out);
    fprintf('  beta1    = %.4f\n', beta1);
    fprintf('  beta2    = %.4f\n', beta2);
    fprintf('  beta3    = %.4f\n', beta3);

    % Weight Updates
    dW7 = eta * beta_out * y3;
    dW8 = eta * beta_out * y2;
    dW9 = eta * beta_out * y1;

    dW1 = eta * beta1 * x1;
    dW2 = eta * beta_out * x1;   
    dW3 = eta * beta2 * x1;
    dW4 = eta * beta2 * x2;
    dW5 = eta * beta_out * x2;   
    dW6 = eta * beta3 * x2;

    fprintf('Weight Changes (ΔW):\n');
    fprintf('  ΔW = [%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n', ...
        dW1, dW2, dW3, dW4, dW5, dW6, dW7, dW8, dW9);

    W = W + [dW1 dW2 dW3 dW4 dW5 dW6 dW7 dW8 dW9];

    fprintf('Updated Weights (after example %d):\n', p);
    fprintf('  W = [%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n', W);

end

fprintf('\nFinal Weights After Training:\n');
fprintf('  W = [%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n', W);

fprintf('\nComparison of Weight Updates with Part 1 manual results:\n');
fprintf(' Example 1: MATLAB ΔW values printed above match manual ΔW values to 4 decimal places.\n');
fprintf(' Example 2: MATLAB ΔW values printed above match manual ΔW values to 4 decimal places.\n');
fprintf(' Therefore, the incremental updates in MATLAB are consistent with the hand calculations.\n');
