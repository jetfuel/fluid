//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/string/piece.h"

#include <sstream>

#include "gtest/gtest.h"

TEST(StringPiece, Construct) {
  {
    paddle::fluid::string::Piece s;
    EXPECT_EQ(NULL, s.data());
    EXPECT_EQ(0U, s.len());
  }
  {
    EXPECT_THROW(paddle::fluid::string::Piece s(NULL, 10000U),
                 std::invalid_argument);
  }
  {
    paddle::fluid::string::Piece s(NULL);
    EXPECT_EQ(0U, s.len());
  }
  {
    std::string a;
    EXPECT_EQ(0U, a.size());
    paddle::fluid::string::Piece s(a);
    EXPECT_EQ(0U, s.len());
  }
}

TEST(StringPiece, CopyAndAssign) {
  paddle::fluid::string::Piece empty;
  EXPECT_EQ(0U, empty.len());

  paddle::fluid::string::Piece a("hello");
  paddle::fluid::string::Piece b = a;
  EXPECT_EQ(b.len(), strlen("hello"));
  EXPECT_EQ(a, b);

  std::string storage("hello");
  paddle::fluid::string::Piece c(storage);
  EXPECT_EQ(a, c);
  EXPECT_NE(a.data(), c.data());
}

TEST(StringPiece, Compare) {
  {
    paddle::fluid::string::Piece a("hello");
    paddle::fluid::string::Piece b("world");
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(a > b);
    EXPECT_FALSE(a >= b);
    EXPECT_LT(Compare(a, b), 0);
    EXPECT_GT(Compare(b, a), 0);
  }
  {
    paddle::fluid::string::Piece a, b;
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a > b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a >= b);
    EXPECT_EQ(0, Compare(a, b));
    EXPECT_EQ(0, Compare(b, a));
  }
}

TEST(StringPiece, ToString) {
  {
    paddle::fluid::string::Piece s;
    EXPECT_EQ(std::string(""), s.ToString());
  }
  {
    paddle::fluid::string::Piece s(NULL);
    EXPECT_EQ(std::string(""), s.ToString());
  }
  {
    paddle::fluid::string::Piece s("hello");
    EXPECT_EQ(std::string("hello"), s.ToString());
  }
}

TEST(StringPiece, HasPrefixSuffix) {
  using paddle::fluid::string::HasPrefix;
  using paddle::fluid::string::HasSuffix;
  {
    paddle::fluid::string::Piece s;
    EXPECT_FALSE(HasPrefix(s, "something"));
    EXPECT_TRUE(HasPrefix(s, ""));
    EXPECT_FALSE(HasSuffix(s, "something"));
    EXPECT_TRUE(HasSuffix(s, ""));
  }
  {
    paddle::fluid::string::Piece s("app");
    EXPECT_TRUE(HasPrefix(s, ""));
    EXPECT_TRUE(HasPrefix(s, "a"));
    EXPECT_TRUE(HasPrefix(s, "ap"));
    EXPECT_TRUE(HasPrefix(s, "app"));

    EXPECT_TRUE(HasSuffix(s, ""));
    EXPECT_TRUE(HasSuffix(s, "p"));
    EXPECT_TRUE(HasSuffix(s, "pp"));
    EXPECT_TRUE(HasSuffix(s, "app"));
  }
}

TEST(StringPiece, SkipPrefixSuffix) {
  using paddle::fluid::string::SkipPrefix;
  using paddle::fluid::string::SkipSuffix;
  {
    paddle::fluid::string::Piece s;
    EXPECT_EQ("", SkipPrefix(s, 0));
    EXPECT_THROW(SkipPrefix(s, 1), std::invalid_argument);

    EXPECT_EQ("", SkipSuffix(s, 0));
    EXPECT_THROW(SkipSuffix(s, 1), std::invalid_argument);
  }
  {
    paddle::fluid::string::Piece s("app");
    EXPECT_EQ("app", SkipPrefix(s, 0));
    EXPECT_EQ("pp", SkipPrefix(s, 1));
    EXPECT_EQ("p", SkipPrefix(s, 2));
    EXPECT_EQ("", SkipPrefix(s, 3));
    EXPECT_THROW(SkipPrefix(s, 4), std::invalid_argument);

    EXPECT_EQ("app", SkipSuffix(s, 0));
    EXPECT_EQ("ap", SkipSuffix(s, 1));
    EXPECT_EQ("a", SkipSuffix(s, 2));
    EXPECT_EQ("", SkipSuffix(s, 3));
    EXPECT_THROW(SkipSuffix(s, 4), std::invalid_argument);
  }
}

TEST(StringPiece, TrimPrefixSuffix) {
  using paddle::fluid::string::TrimPrefix;
  using paddle::fluid::string::TrimSuffix;
  {
    paddle::fluid::string::Piece s;
    EXPECT_EQ("", TrimPrefix(s, ""));
    EXPECT_EQ("", TrimPrefix(s, "something"));

    EXPECT_EQ("", TrimSuffix(s, ""));
    EXPECT_EQ("", TrimSuffix(s, "something"));
  }
  {
    paddle::fluid::string::Piece s("app");
    EXPECT_EQ("app", TrimPrefix(s, ""));
    EXPECT_EQ("pp", TrimPrefix(s, "a"));
    EXPECT_EQ("p", TrimPrefix(s, "ap"));
    EXPECT_EQ("", TrimPrefix(s, "app"));
    EXPECT_EQ("app", TrimPrefix(s, "something"));

    EXPECT_EQ("app", TrimSuffix(s, ""));
    EXPECT_EQ("ap", TrimSuffix(s, "p"));
    EXPECT_EQ("a", TrimSuffix(s, "pp"));
    EXPECT_EQ("", TrimSuffix(s, "app"));
    EXPECT_EQ("app", TrimSuffix(s, "something"));
  }
}

TEST(StringPiece, Contains) {
  using paddle::fluid::string::Contains;
  {
    paddle::fluid::string::Piece s;
    EXPECT_FALSE(Contains(s, ""));
    EXPECT_FALSE(Contains(s, "something"));
  }
  {
    paddle::fluid::string::Piece s("app");
    EXPECT_TRUE(Contains(s, ""));
    EXPECT_TRUE(Contains(s, "a"));
    EXPECT_TRUE(Contains(s, "p"));
    EXPECT_TRUE(Contains(s, "ap"));
    EXPECT_TRUE(Contains(s, "pp"));
    EXPECT_TRUE(Contains(s, "app"));
    EXPECT_FALSE(Contains(s, "something"));
  }
}

TEST(StringPiece, Index) {
  using paddle::fluid::string::Index;
  auto npos = paddle::fluid::string::Piece::npos;
  {
    paddle::fluid::string::Piece s;
    EXPECT_EQ(npos, Index(s, ""));
    EXPECT_EQ(npos, Index(s, "something"));
  }
  {
    paddle::fluid::string::Piece s("app");
    EXPECT_EQ(0U, Index(s, ""));
    EXPECT_EQ(0U, Index(s, "a"));
    EXPECT_EQ(1U, Index(s, "p"));
    EXPECT_EQ(0U, Index(s, "ap"));
    EXPECT_EQ(1U, Index(s, "pp"));
    EXPECT_EQ(0U, Index(s, "app"));
    EXPECT_EQ(npos, Index(s, "something"));
  }
}

TEST(StringPiece, Find) {
  using paddle::fluid::string::Find;
  auto npos = paddle::fluid::string::Piece::npos;
  {
    paddle::fluid::string::Piece s;
    EXPECT_EQ(npos, Find(s, 'a', 0U));
  }
  {
    paddle::fluid::string::Piece s("app");
    EXPECT_EQ(0U, Find(s, 'a', 0U));
    EXPECT_EQ(1U, Find(s, 'p', 0U));
    EXPECT_EQ(1U, Find(s, 'p', 1U));
    EXPECT_EQ(2U, Find(s, 'p', 2U));
    EXPECT_EQ(npos, Find(s, 'z', 2U));
  }
}

TEST(StringPiece, RFind) {
  using paddle::fluid::string::RFind;
  auto npos = paddle::fluid::string::Piece::npos;
  {
    paddle::fluid::string::Piece s;
    EXPECT_EQ(npos, RFind(s, 'a', 0U));
  }
  {
    paddle::fluid::string::Piece s("app");
    EXPECT_EQ(2U, RFind(s, 'p', 2U));
    EXPECT_EQ(0U, RFind(s, 'a', 2U));
    EXPECT_EQ(1U, RFind(s, 'p', 1U));
    EXPECT_EQ(0U, RFind(s, 'a', 0));
    EXPECT_EQ(npos, RFind(s, 'z', 2U));
  }
}

TEST(StringPiece, SubStr) {
  using paddle::fluid::string::SubStr;
  {
    paddle::fluid::string::Piece s;
    EXPECT_EQ("", SubStr(s, 0, 0));
    EXPECT_EQ("", SubStr(s, 0, 1));
    EXPECT_EQ("", SubStr(s, 1, 0));
  }
  {
    paddle::fluid::string::Piece s("app");
    EXPECT_EQ("", SubStr(s, 0, 0));
    EXPECT_EQ("", SubStr(s, 1, 0));
    EXPECT_EQ("", SubStr(s, 2, 0));
    EXPECT_EQ("", SubStr(s, 3, 0));

    EXPECT_EQ("a", SubStr(s, 0, 1));
    EXPECT_EQ("p", SubStr(s, 1, 1));
    EXPECT_EQ("p", SubStr(s, 2, 1));
    EXPECT_EQ("", SubStr(s, 3, 1));

    EXPECT_EQ("ap", SubStr(s, 0, 2));
    EXPECT_EQ("pp", SubStr(s, 1, 2));
    EXPECT_EQ("p", SubStr(s, 2, 2));
    EXPECT_EQ("", SubStr(s, 3, 2));

    EXPECT_EQ("app", SubStr(s, 0, 3));
    EXPECT_EQ("pp", SubStr(s, 1, 3));
    EXPECT_EQ("p", SubStr(s, 2, 3));
    EXPECT_EQ("", SubStr(s, 3, 3));
  }
}

TEST(StringPiece, StreamOutput) {
  using paddle::fluid::string::Piece;

  std::stringstream o;
  o << paddle::fluid::string::Piece();
  EXPECT_EQ("", o.str());

  o << paddle::fluid::string::Piece("hello");
  EXPECT_EQ("hello", o.str());

  o << paddle::fluid::string::Piece();
  EXPECT_EQ("hello", o.str());
}
