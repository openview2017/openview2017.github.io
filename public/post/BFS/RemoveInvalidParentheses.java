/*

301. Remove Invalid Parentheses
Hard


Given a string s that contains parentheses and letters, remove the minimum number of invalid parentheses to make the input string valid.

Return all the possible results. You may return the answer in any order.

 

Example 1:

Input: s = "()())()"
Output: ["(())()","()()()"]
Example 2:

Input: s = "(a)())()"
Output: ["(a())()","(a)()()"]
Example 3:

Input: s = ")("
Output: [""]


*/

public class RemoveInvalidParentheses {
    class Solution {
        public List<String> removeInvalidParentheses(String s) {
          List<String> res = new ArrayList<>();
          
          // sanity check
          if (s == null) return res;
          int minLen = 0;
            
          Set<String> visited = new HashSet<>();
          Queue<String> queue = new LinkedList<>();
          
          // initialize
          queue.add(s);
          visited.add(s);
          
          boolean found = false;
          
          while (!queue.isEmpty()) {
            s = queue.poll();
            
            if (isValid(s)) {
              // found an answer, add to the result
              res.add(s);
              found = true;
            }
          
            if (found) continue;
          
            // generate all possible states
            for (int i = 0; i < s.length(); i++) {
              // we only try to remove left or right paren
              if (s.charAt(i) != '(' && s.charAt(i) != ')') continue;
            
              String t = s.substring(0, i) + s.substring(i + 1);
            
              if (!visited.contains(t)) {
                // for each state, if it's not visited, add it to the queue
                queue.add(t);
                visited.add(t);
              }
            }
          }
          
          return res;
        }
        
        // helper function checks if string s contains valid parantheses
        boolean isValid(String s) {
          int count = 0;
        
          for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') count++;
            if (c == ')' && count-- == 0) return false;
          }
        
          return count == 0;
        }
    }
    /*
    
    1 2 3 4 (5) 6
    / ////|  ....
    7 8 9 10 (work for length of i, not work for all i-1 length)
    */
    
    
    
    class Solution {
        public List<String> removeInvalidParentheses(String s) {
            int lremove = 0;
            int rremove = 0;
            List<Integer> left = new ArrayList<Integer>();
            List<Integer> right = new ArrayList<Integer>();
            List<String> ans = new ArrayList<String>();
            Set<String> cnt = new HashSet<String>();
    
            for (int i = 0; i < s.length(); i++) {
                if (s.charAt(i) == '(') {
                    left.add(i);
                    lremove++;
                } else if (s.charAt(i) == ')') {
                    right.add(i);
                    if (lremove == 0) {
                        rremove++;
                    } else {
                        lremove--;
                    }
                }
            }
    
            int m = left.size();
            int n = right.size();
            List<Integer> maskArr1 = new ArrayList<Integer>();
            List<Integer> maskArr2 = new ArrayList<Integer>();
            for (int i = 0; i < (1 << m); i++) {
                if (Integer.bitCount(i) != lremove) {
                    continue;
                }
                maskArr1.add(i);
            }
            for (int i = 0; i < (1 << n); i++) {
                if (Integer.bitCount(i) != rremove) {
                    continue;
                }
                maskArr2.add(i);
            }
            for (int mask1 : maskArr1) {
                for (int mask2 : maskArr2) {
                    if (checkValid(s, mask1, left, mask2, right)) {
                        cnt.add(recoverStr(s, mask1, left, mask2, right));
                    }
                }
            }
            for (String v : cnt) {
                ans.add(v);
            }
    
            return ans;
        }
    
        private boolean checkValid(String str, int lmask, List<Integer> left, int rmask, List<Integer> right) {
            int pos1 = 0;
            int pos2 = 0;
            int cnt = 0;
    
            for (int i = 0; i < str.length(); i++) {
                if (pos1 < left.size() && i == left.get(pos1)) {
                    if ((lmask & (1 << pos1)) == 0) {
                        cnt++;
                    }
                    pos1++;
                } else if (pos2 < right.size() && i == right.get(pos2)) {
                    if ((rmask & (1 << pos2)) == 0) {
                        cnt--;
                        if (cnt < 0) {
                            return false;
                        }
                    }
                    pos2++;
                }
            }
    
            return cnt == 0;
        }
    
        private String recoverStr(String str, int lmask, List<Integer> left, int rmask, List<Integer> right) {
            StringBuilder sb = new StringBuilder();
            int pos1 = 0;
            int pos2 = 0;
    
            for (int i = 0; i < str.length(); i++) {
                if (pos1 < left.size() && i == left.get(pos1)) {
                    if ((lmask & (1 << pos1)) == 0) {
                        sb.append(str.charAt(i));
                    }
                    pos1++;
                } else if (pos2 < right.size() && i == right.get(pos2)) {
                    if ((rmask & (1 << pos2)) == 0) {
                        sb.append(str.charAt(i));
                    }
                    pos2++;
                } else {
                    sb.append(str.charAt(i));
                }
            }
    
            return sb.toString();
        }
    }
    
        
}
