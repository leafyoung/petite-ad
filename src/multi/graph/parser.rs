//! Small expression parser for `Graph` construction.
//!
//! Supported grammar is intentionally small and predictable:
//! numbers, named inputs, `+ - * / ^`, parentheses, unary minus, and functions
//! `sin`, `cos`, `tan`, `tanh`, `relu`, `exp`, `ln`, `sqrt`, `abs`, `sigmoid`,
//! `softplus`, `log1p_exp`, and `gelu`.

use crate::{AutodiffError, Graph, NodeId, Result};

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

pub(crate) fn parse_expression(expression: &str, input_names: &[&str]) -> Result<Graph> {
    let tokens = tokenize(expression)?;
    let mut parser = Parser {
        tokens,
        pos: 0,
        graph: Graph::new(input_names.len()),
        input_names: input_names.iter().map(|name| (*name).to_string()).collect(),
    };
    for (index, name) in input_names.iter().enumerate() {
        parser.graph.set_input_name(index, *name)?;
    }
    let output = parser.parse_expr()?;
    if parser.pos != parser.tokens.len() {
        return Err(AutodiffError::InvalidGraph {
            reason: "unexpected trailing expression tokens",
        });
    }
    parser.graph.set_output(output)?;
    Ok(parser.graph)
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut chars = input.chars().peekable();
    let mut tokens = Vec::new();
    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '0'..='9' | '.' => {
                let mut text = String::new();
                let mut previous_was_exponent = false;
                while let Some(&next) = chars.peek() {
                    if next.is_ascii_digit() || next == '.' || next == 'e' || next == 'E' {
                        previous_was_exponent = next == 'e' || next == 'E';
                        text.push(next);
                        chars.next();
                    } else if (next == '-' || next == '+') && previous_was_exponent {
                        previous_was_exponent = false;
                        text.push(next);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let value = text
                    .parse::<f64>()
                    .map_err(|_| AutodiffError::InvalidGraph {
                        reason: "invalid numeric literal",
                    })?;
                tokens.push(Token::Number(value));
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut ident = String::new();
                while let Some(&next) = chars.peek() {
                    if next.is_ascii_alphanumeric() || next == '_' {
                        ident.push(next);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Ident(ident));
            }
            '+' => {
                chars.next();
                tokens.push(Token::Plus);
            }
            '-' => {
                chars.next();
                tokens.push(Token::Minus);
            }
            '*' => {
                chars.next();
                tokens.push(Token::Star);
            }
            '/' => {
                chars.next();
                tokens.push(Token::Slash);
            }
            '^' => {
                chars.next();
                tokens.push(Token::Caret);
            }
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            _ => {
                return Err(AutodiffError::InvalidGraph {
                    reason: "unsupported expression character",
                });
            }
        }
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    graph: Graph,
    input_names: Vec<String>,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn bump(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned();
        self.pos += usize::from(token.is_some());
        token
    }

    fn parse_expr(&mut self) -> Result<NodeId> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<NodeId> {
        let mut left = self.parse_mul_div()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.bump();
                    let right = self.parse_mul_div()?;
                    left = self.graph.add(left, right);
                }
                Some(Token::Minus) => {
                    self.bump();
                    let right = self.parse_mul_div()?;
                    left = self.graph.sub(left, right);
                }
                _ => return Ok(left),
            }
        }
    }

    fn parse_mul_div(&mut self) -> Result<NodeId> {
        let mut left = self.parse_pow()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.bump();
                    let right = self.parse_pow()?;
                    left = self.graph.mul(left, right);
                }
                Some(Token::Slash) => {
                    self.bump();
                    let right = self.parse_pow()?;
                    left = self.graph.div(left, right);
                }
                _ => return Ok(left),
            }
        }
    }

    fn parse_pow(&mut self) -> Result<NodeId> {
        let left = self.parse_unary()?;
        if matches!(self.peek(), Some(Token::Caret)) {
            self.bump();
            let right = self.parse_pow()?;
            Ok(self.graph.pow(left, right))
        } else {
            Ok(left)
        }
    }

    fn parse_unary(&mut self) -> Result<NodeId> {
        if matches!(self.peek(), Some(Token::Minus)) {
            self.bump();
            let node = self.parse_unary()?;
            Ok(self.graph.neg(node))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<NodeId> {
        match self.bump() {
            Some(Token::Number(value)) => Ok(self.graph.constant(value)),
            Some(Token::Ident(name)) => {
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.bump();
                    let arg = self.parse_expr()?;
                    if !matches!(self.bump(), Some(Token::RParen)) {
                        return Err(AutodiffError::InvalidGraph {
                            reason: "function call missing closing parenthesis",
                        });
                    }
                    self.apply_function(&name, arg)
                } else {
                    let index = self
                        .input_names
                        .iter()
                        .position(|candidate| candidate == &name)
                        .ok_or(AutodiffError::InvalidGraph {
                            reason: "unknown expression identifier",
                        })?;
                    Ok(self.graph.input(index))
                }
            }
            Some(Token::LParen) => {
                let node = self.parse_expr()?;
                if !matches!(self.bump(), Some(Token::RParen)) {
                    return Err(AutodiffError::InvalidGraph {
                        reason: "expression missing closing parenthesis",
                    });
                }
                Ok(node)
            }
            _ => Err(AutodiffError::InvalidGraph {
                reason: "expected expression primary",
            }),
        }
    }

    fn apply_function(&mut self, name: &str, arg: NodeId) -> Result<NodeId> {
        Ok(match name {
            "sin" => self.graph.sin(arg),
            "cos" => self.graph.cos(arg),
            "tan" => self.graph.tan(arg),
            "tanh" => self.graph.tanh(arg),
            "relu" => self.graph.relu(arg),
            "exp" => self.graph.exp(arg),
            "ln" | "log" => self.graph.ln(arg),
            "sqrt" => self.graph.sqrt(arg),
            "abs" => self.graph.abs(arg),
            "sigmoid" => self.graph.sigmoid(arg),
            "softplus" | "log1p_exp" => self.graph.log1p_exp(arg),
            "gelu" => self.graph.gelu(arg),
            _ => {
                return Err(AutodiffError::InvalidGraph {
                    reason: "unknown expression function",
                });
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::approx_eq_eps as approx_eq;

    // ---- Valid parse + evaluate round trips ----

    #[test]
    fn test_parse_single_input_addition() {
        // f(x) = x + x
        let graph = parse_expression("x + x", &["x"]).unwrap();
        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 6.0, 1e-10));
    }

    #[test]
    fn test_parse_two_variables_mul() {
        // f(x, y) = x * y
        let graph = parse_expression("x * y", &["x", "y"]).unwrap();
        let value = graph.compute(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 6.0, 1e-10));
    }

    #[test]
    fn test_parse_subtraction_and_division() {
        // f(a, b) = (a - b) / b
        let graph = parse_expression("(a - b) / b", &["a", "b"]).unwrap();
        let value = graph.compute(&[5.0, 2.0]).unwrap();
        assert!(approx_eq(value, 1.5, 1e-10));
    }

    #[test]
    fn test_parse_power() {
        // f(x, y) = x ^ y
        let graph = parse_expression("x ^ y", &["x", "y"]).unwrap();
        let value = graph.compute(&[2.0, 3.0]).unwrap();
        assert!(approx_eq(value, 8.0, 1e-10));
    }

    #[test]
    fn test_parse_unary_minus() {
        // f(x) = -x + 5
        let graph = parse_expression("-x + 5", &["x"]).unwrap();
        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 2.0, 1e-10));
    }

    #[test]
    fn test_parse_nested_unary_minus() {
        // f(x) = --x (double negation)
        let graph = parse_expression("--x", &["x"]).unwrap();
        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));
    }

    #[test]
    fn test_parse_numeric_constant() {
        // Use a non-PI approximate constant
        let graph = parse_expression("2.71", &[]).unwrap();
        let value = graph.compute(&[]).unwrap();
        assert!(approx_eq(value, 2.71, 1e-10));
    }

    #[test]
    fn test_parse_scientific_notation() {
        // f() = 1e-3
        let graph = parse_expression("1e-3", &[]).unwrap();
        let value = graph.compute(&[]).unwrap();
        assert!(approx_eq(value, 1e-3, 1e-10));
    }

    #[test]
    fn test_parse_scientific_notation_positive_exponent() {
        // f() = 2E+5
        let graph = parse_expression("2E+5", &[]).unwrap();
        let value = graph.compute(&[]).unwrap();
        assert!(approx_eq(value, 2e5, 1e-8));
    }

    #[test]
    fn test_parse_function_sin() {
        // f(x) = sin(x)
        let graph = parse_expression("sin(x)", &["x"]).unwrap();
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 0.0, 1e-10));
    }

    #[test]
    fn test_parse_function_cos() {
        // f(x) = cos(x)
        let graph = parse_expression("cos(x)", &["x"]).unwrap();
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 1.0, 1e-10));
    }

    #[test]
    fn test_parse_function_tan() {
        // f(x) = tan(x)
        let graph = parse_expression("tan(x)", &["x"]).unwrap();
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 0.0, 1e-10));
    }

    #[test]
    fn test_parse_function_tanh() {
        // f(x) = tanh(x)
        let graph = parse_expression("tanh(x)", &["x"]).unwrap();
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 0.0, 1e-10));
    }

    #[test]
    fn test_parse_function_relu() {
        // f(x) = relu(x)
        let graph = parse_expression("relu(x)", &["x"]).unwrap();
        let value_pos = graph.compute(&[2.0]).unwrap();
        assert!(approx_eq(value_pos, 2.0, 1e-10));
    }

    #[test]
    fn test_parse_function_relu_negative() {
        let graph = parse_expression("relu(x)", &["x"]).unwrap();
        let value_neg = graph.compute(&[-2.0]).unwrap();
        assert!(approx_eq(value_neg, 0.0, 1e-10));
    }

    #[test]
    fn test_parse_function_exp() {
        // f(x) = exp(x)
        let graph = parse_expression("exp(x)", &["x"]).unwrap();
        let value = graph.compute(&[1.0]).unwrap();
        assert!(approx_eq(value, std::f64::consts::E, 1e-10));
    }

    #[test]
    fn test_parse_function_ln() {
        // f(x) = ln(x)
        let graph = parse_expression("ln(x)", &["x"]).unwrap();
        let value = graph.compute(&[std::f64::consts::E]).unwrap();
        assert!(approx_eq(value, 1.0, 1e-10));
    }

    #[test]
    fn test_parse_function_log_alias() {
        // f(x) = log(x)  (alias for ln)
        let graph = parse_expression("log(x)", &["x"]).unwrap();
        let value = graph.compute(&[std::f64::consts::E]).unwrap();
        assert!(approx_eq(value, 1.0, 1e-10));
    }

    #[test]
    fn test_parse_function_sqrt() {
        // f(x) = sqrt(x)
        let graph = parse_expression("sqrt(x)", &["x"]).unwrap();
        let value = graph.compute(&[4.0]).unwrap();
        assert!(approx_eq(value, 2.0, 1e-10));
    }

    #[test]
    fn test_parse_function_abs() {
        // f(x) = abs(x)
        let graph = parse_expression("abs(x)", &["x"]).unwrap();
        let value = graph.compute(&[-3.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));
    }

    #[test]
    fn test_parse_function_sigmoid() {
        // f(x) = sigmoid(x)
        let graph = parse_expression("sigmoid(x)", &["x"]).unwrap();
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 0.5, 1e-10));
    }

    #[test]
    fn test_parse_function_softplus() {
        // f(x) = softplus(x)  (alias for log1p_exp)
        let graph = parse_expression("softplus(x)", &["x"]).unwrap();
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 2.0_f64.ln(), 1e-6));
    }

    #[test]
    fn test_parse_function_log1p_exp() {
        // f(x) = log1p_exp(x)
        let graph = parse_expression("log1p_exp(x)", &["x"]).unwrap();
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 2.0_f64.ln(), 1e-6));
    }

    #[test]
    fn test_parse_function_gelu() {
        // f(x) = gelu(x)
        let graph = parse_expression("gelu(x)", &["x"]).unwrap();
        let value = graph.compute(&[0.0]).unwrap();
        assert!(approx_eq(value, 0.0, 1e-10));
    }

    #[test]
    fn test_parse_complex_expression() {
        // f(x, y) = sin(x) * (x + y)
        let graph = parse_expression("sin(x) * (x + y)", &["x", "y"]).unwrap();
        let value = graph.compute(&[0.6, 1.4]).unwrap();
        let expected = 0.6_f64.sin() * (0.6 + 1.4);
        assert!(approx_eq(value, expected, 1e-10));
    }

    #[test]
    fn test_parse_chained_functions() {
        // f(x) = sin(cos(exp(x)))
        let graph = parse_expression("sin(cos(exp(x)))", &["x"]).unwrap();
        let value = graph.compute(&[0.5]).unwrap();
        let expected = 0.5_f64.exp().cos().sin();
        assert!(approx_eq(value, expected, 1e-10));
    }

    #[test]
    fn test_parse_whitespace_handling() {
        // f(x, y) = x+y with various whitespace
        let graph = parse_expression("  x  +  y  ", &["x", "y"]).unwrap();
        let value = graph.compute(&[1.0, 2.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));
    }

    #[test]
    fn test_parse_operator_precedence() {
        // 2 + 3 * 4 = 14 (not 20)
        let graph = parse_expression("2 + 3 * 4", &[]).unwrap();
        let value = graph.compute(&[]).unwrap();
        assert!(approx_eq(value, 14.0, 1e-10));
    }

    #[test]
    fn test_parse_deeply_nested_parens() {
        // f(x) = ((x + 1))
        let graph = parse_expression("((x + 1))", &["x"]).unwrap();
        let value = graph.compute(&[2.0]).unwrap();
        assert!(approx_eq(value, 3.0, 1e-10));
    }

    #[test]
    fn test_parse_mul_div_precedence() {
        // 6 / 2 * 3 = 9 (left-to-right)
        let graph = parse_expression("6 / 2 * 3", &[]).unwrap();
        let value = graph.compute(&[]).unwrap();
        assert!(approx_eq(value, 9.0, 1e-10));
    }

    #[test]
    fn test_parse_add_sub_precedence() {
        // 10 - 3 - 2 = 5 (left-to-right)
        let graph = parse_expression("10 - 3 - 2", &[]).unwrap();
        let value = graph.compute(&[]).unwrap();
        assert!(approx_eq(value, 5.0, 1e-10));
    }

    #[test]
    fn test_parse_float_literal() {
        // f() = 0.5
        let graph = parse_expression("0.5", &[]).unwrap();
        let value = graph.compute(&[]).unwrap();
        assert!(approx_eq(value, 0.5, 1e-10));
    }

    #[test]
    fn test_parse_dot_literal() {
        // f() = .5 (starting with dot)
        let graph = parse_expression(".5", &[]).unwrap();
        let value = graph.compute(&[]).unwrap();
        assert!(approx_eq(value, 0.5, 1e-10));
    }

    #[test]
    fn test_parse_negative_times_variable() {
        // f(x) = -2 * x
        let graph = parse_expression("-2 * x", &["x"]).unwrap();
        let value = graph.compute(&[3.0]).unwrap();
        assert!(approx_eq(value, -6.0, 1e-10));
    }

    #[test]
    fn test_parse_power_right_associative() {
        // 2 ^ 3 ^ 2 = 2^(3^2) = 512
        let graph = parse_expression("2 ^ 3 ^ 2", &[]).unwrap();
        let value = graph.compute(&[]).unwrap();
        assert!(approx_eq(value, 512.0, 1e-10));
    }

    // ---- Error cases ----

    #[test]
    fn test_parse_error_unsupported_character() {
        let result = parse_expression("x # y", &["x", "y"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidGraph { .. }));
    }

    #[test]
    fn test_parse_error_unknown_identifier() {
        let result = parse_expression("x + z", &["x", "y"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AutodiffError::InvalidGraph { reason } if reason.contains("unknown expression identifier"))
        );
    }

    #[test]
    fn test_parse_error_unknown_function() {
        let result = parse_expression("foo(x)", &["x"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AutodiffError::InvalidGraph { reason } if reason.contains("unknown expression function"))
        );
    }

    #[test]
    fn test_parse_error_function_missing_closing_paren() {
        let result = parse_expression("sin(x", &["x"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AutodiffError::InvalidGraph { reason } if reason.contains("function call missing closing parenthesis"))
        );
    }

    #[test]
    fn test_parse_error_expression_missing_closing_paren() {
        let result = parse_expression("(x + y", &["x", "y"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AutodiffError::InvalidGraph { reason } if reason.contains("expression missing closing parenthesis"))
        );
    }

    #[test]
    fn test_parse_error_trailing_tokens() {
        let result = parse_expression("x + y z", &["x", "y"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AutodiffError::InvalidGraph { reason } if reason.contains("unexpected trailing expression tokens"))
        );
    }

    #[test]
    fn test_parse_error_expected_primary() {
        // Just a plus sign with nothing before it
        let result = parse_expression("+", &["x"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_invalid_numeric_literal() {
        // "1e" is not a valid float
        let result = parse_expression("1e", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_expression() {
        let result = parse_expression("", &[]);
        assert!(result.is_err());
    }
}
