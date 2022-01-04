/*
125. Valid Palindrome
Easy

A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

 

Example 1:

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.



*/
class IsPalindrome {
    public boolean isPalindrome(String s) {
        if (s.length() <= 1) {
            return true;
        }
        int left = 0, right = s.length() - 1;
        int lChar = s.charAt(left), rChar = s.charAt(right);
        while (left < right) {
            lChar = s.charAt(left);
            rChar = s.charAt(right);
            while (left < right && !Character.isLetterOrDigit(lChar)) { // cannot be <=, it won't work for ",."
                lChar = s.charAt(++left);
            }
            while (left < right && !Character.isLetterOrDigit(rChar)) {
                rChar = s.charAt(--right);
            }
            if (Character.toLowerCase(lChar) != Character.toLowerCase(rChar)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}