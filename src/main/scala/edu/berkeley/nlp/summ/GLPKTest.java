package edu.berkeley.nlp.summ;

import org.gnu.glpk.GLPK;
import org.gnu.glpk.glp_prob;

/**
 * Small class to let you easily test whether your Java library path settings
 * are correct. Run with
 * 
 * -Djava.library.path="<existing path>:<path to libglpk_java libraries"
 * 
 * which may be located in /usr/local/lib/jni
 * 
 * @author gdurrett
 *
 */
public class GLPKTest {
  
  public static void main(String[] args) {
    String javaLibPath = System.getProperty("java.library.path");
    System.out.println(javaLibPath);
    System.out.println("is your Java library path");
    
    if (args.length == 1 && args[0].equals("noglpk")) {
      
    } else {
      glp_prob lp = GLPK.glp_create_prob();
      System.out.println("GLPK successful!"); 
    }
  }
}
