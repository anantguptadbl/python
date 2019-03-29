# REINFORCEMENT ALGORITHMS
Reinforcement algorithms are powerful algorithms that help us find the next best state

## CARTPOLE_V1
This uses the cartpole GYM of OpenAI Gym
I feel that OpenAI is the best place to learn reinforcement algorithms in a practical fashion

## TIC TAC TOE Simple
I felt that i am able to make a machine learn to play TIC TAC TOE, i might then be able to teach it greater things :)

## TIC TAC TOE Q Algorithm 1
Make a machine learn to play TIC TAC TOE using the Q algorithm </br>
Inspiration for the Q algorithm came from this algorithm </br>
http://mnemstudio.org/path-finding-q-learning-tutorial.htm

Q(S,A) = Q(S,A) + R(S,A) + gamma * max( Q(A,A+1) )

## TIC TAC TOE Q Algorithm 2

Q(S,A) = Q(S,A) + alpha * ( R(S,A) + gamma * max(Q(A,A+1)) - Q(S,A) )
