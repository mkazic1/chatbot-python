import re


def message_probability(user_message, recognised_words, required_words=[]):
    certainty = 0
    contains_required_words = True

    # Counts how many words are present in each predefined message
    for word in user_message:
        if word in recognised_words:
            certainty += 1

    # Calculates the percent of recognised words in a user message
    percentage = float(certainty) / float(len(recognised_words))

    # Checks that the required words are in the string
    for word in required_words:
        if word not in user_message:
            contains_required_words = False
            break

    # Checking if required words constraint is fullfiled
    if contains_required_words:
        return int(percentage * 100)
    else:
        return 0

def check_for_responses(message):
    highest_prob_list = {}

    # Add responses to dictionary
    def response(bot_response, list_of_words, required_words=[]):
        nonlocal highest_prob_list
        highest_prob_list[bot_response] = message_probability(message, list_of_words, required_words)

    # Responses -------------------------------------------------------------------------------------------------------
    response('Hello!', ['hello', 'hi', 'hey', 'sup', 'heyo'])
    response('See you!', ['bye', 'goodbye'])
    response('I\'m doing fine, and you?', ['how', 'are', 'you', 'doing'], required_words=['how'])
    response('You\'re welcome!', ['thank', 'thanks'])
    response('Thank you!', ['i', 'love', 'code', 'palace'], required_words=['code', 'palace'])

    best_match = max(highest_prob_list, key=highest_prob_list.get)
    print(highest_prob_list)
    print(f'Best match = {best_match} | Score: {highest_prob_list[best_match]}')

    return best_match if highest_prob_list[best_match] > 1 else "Sorry, couldnt answer"

# Get the response
def get_response(user_input):
    split_message = re.split(r'\s+|[,;?!.-]\s*', user_input.lower())
    response = check_for_responses(split_message)
    return response

# Testing the response system
while True:
    print('Bot: ' + get_response(input('You: ')))



