<?php
namespace Foo {
function curly( $x ) {
	$a = "v={$x->id}";
	$b = "w=${x}";
	return $a . $b;
}
interface Shape {
	public function area(): float;
	public function name(): string;
}
abstract class Base {
	abstract protected function build( $a );
	public function after() {
		return 1;
	}
}
function make() {
	$o = new class {
		public function inner() {
			return 2;
		}
	};
	return $o;
}
class Real {
	public function m( $id ) {
		$f = function ( $y ) { return $y; };
		return $f( $id );
	}
}
}
namespace {
function global_fn() {
	return 3;
}
}
