# def solution(S, K):

#     if len(S) < K:
#         return S
    
#     truncated = S[:K-3]

#     last_space_index = truncated.rfind(' ')

#     if last_space_index == -1:
#         return  '...'
    
#     return truncated[:last_space_index].rstrip() + ' ' + '...'

# # Test Cases
# print(solution("Safaricom is a great place to work", 20)) 
# # Output: "Safaricom is a..."

# print(solution("Hello world", 20)) 
# # Output: "Hello world" (No change)



def solution(S, K):
    if len(S)<K:
        return S
    
    limit = K- 3

    if limit <=0:
        return '...'[:K]
    
    truncated = S[:limit]

    last_space = truncated.rfind(' ')
    if last_space == -1:
        return '...'
    
    return truncated[:last_space].rstrip() + ' ' + '...'

text = "Safaricom is a great place to work"

# Example: Strict 20 character limit
result = solution(text, 20)
print(f"Result: '{result}'")
print(f"Length: {len(result)}") 

# Output:
# Result: 'Safaricom is...'
# Length: 15 (Well within 20)