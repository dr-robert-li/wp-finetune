<?php
// Unit tests. Must be skipped (test).
class Test_Registrations extends WP_UnitTestCase {
	public function test_count_is_zero_for_new_event() {
		$this->assertSame( array(), ser_get_registrations_for_event( 0 ) );
	}
}
