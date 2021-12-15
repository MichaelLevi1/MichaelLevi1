## Michael Leventeris - 15/12/2021
## Number Guessing Game

import random

print("Welcome to Guessing Game!!")

Range_Upper = (input("What range do you want to guess from: "))
# Check whether input is a positive number
while Range_Upper.isdigit() == False:
    print("Sorry, that is not a positive integer value.")
    Range_Upper = input("Please try again: ")

Range_Upper = int(Range_Upper)
Range_Lower = 0
# Generating random number
number = random.randint(Range_Lower,Range_Upper)

Counter = 0
# Runs while loop until the number is achieved, and recording the amount of terms
while True:
    Guess = input("Guess a number between %d and %d: " %(Range_Lower,Range_Upper))
    # Check whether input is a positive number
    while Guess.isdigit() == False:
        print("Sorry, that is not a positive integer value.")
        Guess = input("Please try again: ")
    Guess = int(Guess)

    # Correct Number
    if Guess == number:
        print("Congratulations, the number was %d. It took you %d guesses" %(Guess, Counter))
        break
    # Not in Range
    elif Guess > Range_Upper or Guess < Range_Lower:
        print("Sorry that is not within the range, try again.")
    else:
        # Number is larger than guess
        if Guess < number:
            Range_Lower = Guess
            print('The number is larger than %d' %(Guess))
        # Number is less than guess
        else: 
            Range_Upper = Guess
            print('The number is smaller than %d' %(Guess))
        Counter += 1
